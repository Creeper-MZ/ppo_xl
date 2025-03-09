import os
import torch
from torch import optim
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from PIL import Image
import numpy as np
import random
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime

from reward_model import AestheticRewardModel

class PPOTrainer:
    """
    PPO Trainer for SDXL, optimizing for aesthetic quality based on a reward model
    """
    
    def __init__(self, 
                 reward_model_path="output/final_model.pth",
                 pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
                 learning_rate=1e-6,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 adam_weight_decay=1e-2,
                 adam_epsilon=1e-8,
                 max_grad_norm=1.0,
                 ppo_epochs=4,
                 ppo_clip_range=0.2,
                 vf_coef=0.5,
                 entropy_coef=0.01,
                 output_dir="output/sdxl_rl",
                 device=None,
                 seed=42):
        """
        Initialize the PPO trainer for SDXL
        
        Args:
            reward_model_path: Path to the aesthetic reward model weights
            pretrained_model_name_or_path: Path to pretrained SDXL model
            learning_rate: Learning rate for optimization
            adam_beta1: Adam beta1 parameter
            adam_beta2: Adam beta2 parameter
            adam_weight_decay: Weight decay for AdamW
            adam_epsilon: Epsilon for AdamW
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per batch
            ppo_clip_range: PPO clip range
            vf_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            output_dir: Directory to save outputs
            device: Device to run training on
            seed: Random seed
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Configure output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
        
        # Load reward model
        print("Loading aesthetic reward model...")
        self.reward_model = AestheticRewardModel(model_path=reward_model_path, device=self.device)
        
        # Load SDXL model
        print(f"Loading SDXL model from {pretrained_model_name_or_path}...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder",
        ).to(self.device)
        
        # We'll be training only the UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        ).to(self.device)
        
        # Create SDXL pipeline for sampling
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=self.unet,
            text_encoder=self.text_encoder,
            safety_checker=None,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        
        # Store original weights for KL divergence
        self.original_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        ).to(self.device)
        self.original_unet.requires_grad_(False)
        
        # Set up optimizer only for UNet
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
        
        # Store hyperparameters
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_range = ppo_clip_range
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        
        print("PPO Trainer initialization complete")
    
    def generate_images(self, prompts, batch_size=4, num_steps=50, guidance_scale=7.5):
        """
        Generate images from the current SDXL model
        
        Args:
            prompts: List of text prompts
            batch_size: Batch size for generation
            num_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            List of generated PIL Images
        """
        all_images = []
        
        # Generate images in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate images
            with torch.no_grad():
                output = self.pipeline(
                    batch_prompts,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    output_type="pil",
                )
            
            all_images.extend(output.images)
        
        return all_images
    
    def compute_kl_divergence(self, noise, timesteps, prompt_embeddings):
        """
        Compute KL divergence between current and original model
        
        Args:
            noise: Noise tensor
            timesteps: Timestep tensor
            prompt_embeddings: Prompt embedding tensor
            
        Returns:
            KL divergence loss
        """
        # Get output from current model
        with torch.enable_grad():
            current_output = self.unet(noise, timesteps, prompt_embeddings).sample
        
        # Get output from original model
        with torch.no_grad():
            original_output = self.original_unet(noise, timesteps, prompt_embeddings).sample
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(current_output, dim=1),
            F.softmax(original_output, dim=1),
            reduction="batchmean"
        )
        
        return kl_loss
    
    def train_step(self, prompts, kl_weight=0.1, num_rollouts=10):
        """
        Perform one training step
        
        Args:
            prompts: List of text prompts
            kl_weight: Weight for KL divergence loss
            num_rollouts: Number of rollouts per prompt
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        all_rewards = []
        all_scores = []
        
        # Generate images and compute rewards
        images = self.generate_images(prompts)
        rewards, scores = self.reward_model.batch_rewards(images)
        
        all_rewards.extend(rewards)
        all_scores.extend(scores)
        
        avg_reward = np.mean(rewards)
        avg_score = np.mean(scores)
        
        # Save sample images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (image, reward, score) in enumerate(zip(images[:5], rewards[:5], scores[:5])):
            image.save(os.path.join(self.output_dir, "samples", f"{timestamp}_{i}_score_{score:.1f}_reward_{reward:.4f}.png"))
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Create a batch of denoising operations for PPO
            for rollout_idx in range(num_rollouts):
                # Sample random noise and timesteps for each prompt
                noise = torch.randn((len(prompts), 4, 128, 128)).to(self.device)
                timesteps = torch.randint(0, 1000, (len(prompts),)).long().to(self.device)
                
                # Encode prompts
                with torch.no_grad():
                    prompt_embeddings = self.pipeline.encode_prompt(prompts)
                    if isinstance(prompt_embeddings, tuple):
                        prompt_embeddings = prompt_embeddings[0]  # Get text embeddings
                
                # Compute KL divergence loss
                kl_loss = self.compute_kl_divergence(noise, timesteps, prompt_embeddings)
                
                # Use the average reward as our RL advantage signal
                # In a more sophisticated setup, we would compute advantages properly
                advantage = torch.tensor(avg_reward).to(self.device)
                
                # Final loss
                loss = -advantage * 10.0 + kl_weight * kl_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Record metrics
        metrics["avg_reward"] = avg_reward
        metrics["avg_score"] = avg_score
        metrics["max_score"] = max(scores)
        metrics["min_score"] = min(scores)
        metrics["kl_loss"] = kl_loss.item()
        
        return metrics, images[:5]  # Return top 5 images for visualization
    
    def save_checkpoint(self, step):
        """
        Save model checkpoint
        
        Args:
            step: Current training step
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save UNet
        self.unet.save_pretrained(os.path.join(checkpoint_dir, "unet"))
        
        # Save full pipeline
        self.pipeline.save_pretrained(checkpoint_dir)
        
        print(f"Saved checkpoint at step {step}")
    
    def train(self, 
              prompt_file, 
              num_steps=1000, 
              log_interval=10, 
              save_interval=100,
              kl_weight=0.1,
              use_wandb=False,
              wandb_project="sdxl-rl",
              wandb_entity=None,
              wandb_run_name=None):
        """
        Train the model
        
        Args:
            prompt_file: File containing prompts, one per line
            num_steps: Number of training steps
            log_interval: Interval for logging metrics
            save_interval: Interval for saving checkpoints
            kl_weight: Weight for KL divergence loss
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_entity: W&B entity name
            wandb_run_name: W&B run name
        """
        # Load prompts
        with open(prompt_file, "r") as f:
            all_prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(all_prompts)} prompts from {prompt_file}")
        
        # Initialize W&B
        if use_wandb:
            if wandb_run_name is None:
                wandb_run_name = f"sdxl-rl-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config={
                    "num_steps": num_steps,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "ppo_epochs": self.ppo_epochs,
                    "ppo_clip_range": self.ppo_clip_range,
                    "kl_weight": kl_weight,
                    "seed": self.seed,
                }
            )
        
        # Training loop
        for step in tqdm(range(1, num_steps + 1)):
            # Sample random prompts
            batch_size = min(4, len(all_prompts))
            prompts = random.sample(all_prompts, batch_size)
            
            # Perform training step
            metrics, sample_images = self.train_step(prompts, kl_weight=kl_weight)
            
            # Log metrics
            if step % log_interval == 0:
                print(f"Step {step}/{num_steps}, Metrics: {metrics}")
                
                if use_wandb:
                    log_dict = {f"train/{k}": v for k, v in metrics.items()}
                    
                    # Log sample images
                    for i, img in enumerate(sample_images):
                        log_dict[f"samples/image_{i}"] = wandb.Image(
                            img, 
                            caption=f"Prompt: {prompts[i][:50]}...")
                    
                    wandb.log(log_dict, step=step)
            
            # Save checkpoint
            if step % save_interval == 0 or step == num_steps:
                self.save_checkpoint(step)
        
        # Final checkpoint
        self.save_checkpoint("final")
        
        if use_wandb:
            wandb.finish()
        
        print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SDXL with reinforcement learning")
    parser.add_argument("--reward_model_path", type=str, default="output/final_model.pth",
                        help="Path to the aesthetic reward model weights")
    parser.add_argument("--sd_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to pretrained SDXL model")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="File containing prompts, one per line")
    parser.add_argument("--output_dir", type=str, default="output/sdxl_rl",
                        help="Directory to save outputs")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of training steps")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="Weight for KL divergence loss")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="sdxl-rl",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = PPOTrainer(
        reward_model_path=args.reward_model_path,
        pretrained_model_name_or_path=args.sd_model_path,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    trainer.train(
        prompt_file=args.prompt_file,
        num_steps=args.num_steps,
        kl_weight=args.kl_weight,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )