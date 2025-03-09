import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import random

from reward_model import AestheticRewardModel

def load_model(model_path, dtype=torch.float16):
    """Load a SDXL model from the given path"""
    print(f"Loading model from {model_path}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True if os.path.exists(os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")) else False,
    ).to("cuda")
    return pipeline

def generate_comparison(
    base_model_path,
    rl_model_path,
    prompts,
    output_dir,
    reward_model_path,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42
):
    """Generate comparison images between base and RL-finetuned models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Load models
    base_pipeline = load_model(base_model_path)
    rl_pipeline = load_model(rl_model_path)
    
    # Load reward model
    reward_model = AestheticRewardModel(model_path=reward_model_path)
    
    # Track metrics
    metrics = {
        "base": {"scores": [], "rewards": []},
        "rl": {"scores": [], "rewards": []}
    }
    
    # Generate comparison for each prompt
    for i, prompt in enumerate(tqdm(prompts, desc="Generating comparisons")):
        # Set the same random seed for both generations
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        
        # Generate with base model
        base_image = base_pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Generate with RL model
        rl_image = rl_pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Get scores and rewards
        base_reward, base_score = reward_model.get_reward(base_image)
        rl_reward, rl_score = reward_model.get_reward(rl_image)
        
        # Save metrics
        metrics["base"]["scores"].append(base_score)
        metrics["base"]["rewards"].append(base_reward)
        metrics["rl"]["scores"].append(rl_score)
        metrics["rl"]["rewards"].append(rl_reward)
        
        # Create comparison image
        comparison = Image.new('RGB', (base_image.width * 2, base_image.height + 100), color='white')
        comparison.paste(base_image, (0, 0))
        comparison.paste(rl_image, (base_image.width, 0))
        
        # Add text with scores
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, base_image.height + 10), f"Base SDXL: Score {base_score:.1f}, Reward {base_reward:.4f}", fill="black", font=font)
        draw.text((base_image.width + 10, base_image.height + 10), f"RL SDXL: Score {rl_score:.1f}, Reward {rl_reward:.4f}", fill="black", font=font)
        draw.text((10, base_image.height + 50), f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}", fill="black", font=font)
        
        # Save comparison image
        comparison_path = os.path.join(output_dir, f"comparison_{i:03d}.png")
        comparison.save(comparison_path)
    
    # Create summary report
    create_summary_report(metrics, prompts, output_dir)
    
    return metrics

def create_summary_report(metrics, prompts, output_dir):
    """Create a summary report with metrics and visualizations"""
    
    # Calculate statistics
    base_avg_score = np.mean(metrics["base"]["scores"])
    rl_avg_score = np.mean(metrics["rl"]["scores"])
    base_avg_reward = np.mean(metrics["base"]["rewards"])
    rl_avg_reward = np.mean(metrics["rl"]["rewards"])
    
    # Score improvements
    score_improvement = rl_avg_score - base_avg_score
    reward_improvement = rl_avg_reward - base_avg_reward
    
    # Count wins
    score_wins = sum(1 for rl_score, base_score in zip(metrics["rl"]["scores"], metrics["base"]["scores"]) if rl_score > base_score)
    reward_wins = sum(1 for rl_reward, base_reward in zip(metrics["rl"]["rewards"], metrics["base"]["rewards"]) if rl_reward > base_reward)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Score comparison
    plt.subplot(2, 2, 1)
    plt.bar(["Base SDXL", "RL SDXL"], [base_avg_score, rl_avg_score])
    plt.title("Average Aesthetic Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Reward comparison
    plt.subplot(2, 2, 2)
    plt.bar(["Base SDXL", "RL SDXL"], [base_avg_reward, rl_avg_reward])
    plt.title("Average Reward")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Score distribution
    plt.subplot(2, 2, 3)
    plt.hist(metrics["base"]["scores"], alpha=0.5, label="Base SDXL", bins=15)
    plt.hist(metrics["rl"]["scores"], alpha=0.5, label="RL SDXL", bins=15)
    plt.axvspan(200, 500, alpha=0.2, color='green', label='Target Range')
    plt.title("Score Distribution")
    plt.xlabel("Aesthetic Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Reward distribution
    plt.subplot(2, 2, 4)
    plt.hist(metrics["base"]["rewards"], alpha=0.5, label="Base SDXL", bins=15)
    plt.hist(metrics["rl"]["rewards"], alpha=0.5, label="RL SDXL", bins=15)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_metrics.png"))
    
    # Create text report
    with open(os.path.join(output_dir, "comparison_report.txt"), "w") as f:
        f.write("SDXL Reinforcement Learning Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Comparison Metrics:\n")
        f.write(f"Base SDXL Average Score: {base_avg_score:.2f}\n")
        f.write(f"RL SDXL Average Score: {rl_avg_score:.2f}\n")
        f.write(f"Score Improvement: {score_improvement:.2f} ({(score_improvement/base_avg_score)*100:.1f}%)\n\n")
        
        f.write(f"Base SDXL Average Reward: {base_avg_reward:.4f}\n")
        f.write(f"RL SDXL Average Reward: {rl_avg_reward:.4f}\n")
        f.write(f"Reward Improvement: {reward_improvement:.4f} ({(reward_improvement/base_avg_reward)*100:.1f}%)\n\n")
        
        f.write(f"RL model wins (higher score): {score_wins}/{len(prompts)} ({score_wins/len(prompts)*100:.1f}%)\n")
        f.write(f"RL model wins (higher reward): {reward_wins}/{len(prompts)} ({reward_wins/len(prompts)*100:.1f}%)\n\n")
        
        f.write("Per-Prompt Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Prompt':<50} {'Base Score':<12} {'RL Score':<12} {'Base Reward':<12} {'RL Reward':<12}\n")
        f.write("-" * 50 + "\n")
        
        for i, prompt in enumerate(prompts):
            prompt_short = prompt[:47] + "..." if len(prompt) > 47 else prompt
            f.write(f"{prompt_short:<50} {metrics['base']['scores'][i]:<12.2f} {metrics['rl']['scores'][i]:<12.2f} "
                    f"{metrics['base']['rewards'][i]:<12.4f} {metrics['rl']['rewards'][i]:<12.4f}\n")
    
    print(f"Created summary report in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare base SDXL and RL-finetuned SDXL")
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to base SDXL model")
    parser.add_argument("--rl_model_path", type=str, required=True,
                        help="Path to RL-finetuned SDXL model")
    parser.add_argument("--reward_model_path", type=str, default="output/final_model.pth",
                        help="Path to the aesthetic reward model weights")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="File containing test prompts (one per line)")
    parser.add_argument("--num_prompts", type=int, default=20,
                        help="Number of prompts to use for comparison")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts_file, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    
    # Sample prompts if needed
    if len(all_prompts) > args.num_prompts:
        random.seed(args.seed)
        prompts = random.sample(all_prompts, args.num_prompts)
    else:
        prompts = all_prompts
    
    print(f"Using {len(prompts)} prompts for comparison")
    
    # Generate comparisons
    generate_comparison(
        base_model_path=args.base_model_path,
        rl_model_path=args.rl_model_path,
        prompts=prompts,
        output_dir=args.output_dir,
        reward_model_path=args.reward_model_path,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

if __name__ == "__main__":
    main()