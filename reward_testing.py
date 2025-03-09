import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import StableDiffusionXLPipeline

from reward_model import AestheticRewardModel

def plot_reward_function(reward_model, output_path="reward_function.png"):
    """Plot the reward function curve"""
    scores = np.linspace(0, 1000, 1000)
    rewards = []
    
    for score in scores:
        # Use the reward function directly
        if score < 200:
            # Linear increase from 0 to 1 as score goes from 0 to 200
            reward = score / 200.0
        elif 200 <= score <= 500:
            # Maximum reward (1.0) for scores in the target range
            reward = 1.0
        else:
            # Exponential decay for scores above 500
            reward = np.exp(-0.002 * (score - 500))
        
        rewards.append(reward)
    
    plt.figure(figsize=(10, 6))
    plt.plot(scores, rewards)
    plt.title("Aesthetic Reward Function")
    plt.xlabel("Aesthetic Score (0-1000)")
    plt.ylabel("Reward Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the 200-500 range
    plt.axvspan(200, 500, alpha=0.2, color='green', label='Target Range (200-500)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved reward function plot to {output_path}")

def test_reward_model(reward_model, images_dir):
    """Test the reward model on a directory of images"""
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    results = []
    
    for image_file in tqdm(image_files, desc="Evaluating images"):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        reward, score = reward_model.get_reward(image)
        
        results.append({
            'file': image_file,
            'score': score,
            'reward': reward
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print results
    print("\nImage Evaluation Results:")
    print("-" * 60)
    print(f"{'Image':<30} {'Aesthetic Score':<15} {'Reward':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['file']:<30} {result['score']:<15.2f} {result['reward']:<10.4f}")
    
    print("-" * 60)
    print(f"Average Score: {np.mean([r['score'] for r in results]):.2f}")
    print(f"Average Reward: {np.mean([r['reward'] for r in results]):.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot scores vs rewards
    plt.subplot(2, 1, 1)
    plt.scatter([r['score'] for r in results], [r['reward'] for r in results], alpha=0.7)
    plt.title("Aesthetic Scores vs Rewards")
    plt.xlabel("Aesthetic Score")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot score distribution
    plt.subplot(2, 1, 2)
    plt.hist([r['score'] for r in results], bins=20, alpha=0.7)
    plt.title("Distribution of Aesthetic Scores")
    plt.xlabel("Aesthetic Score")
    plt.ylabel("Count")
    plt.axvspan(200, 500, alpha=0.2, color='green', label='Target Range')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "evaluation_results.png"))
    print(f"Saved evaluation visualization to {os.path.join(images_dir, 'evaluation_results.png')}")

def generate_test_images(output_dir, num_images=20, prompts=None):
    """Generate test images using SDXL"""
    os.makedirs(output_dir, exist_ok=True)
    
    if prompts is None or len(prompts) == 0:
        # Default prompts with varying complexity and style
        prompts = [
            "A beautiful mountain landscape with a lake at sunset",
            "Abstract digital art with vibrant colors",
            "Portrait of a smiling woman, photorealistic",
            "Simple sketch of a cat",
            "Cyberpunk cityscape at night with neon lights",
            "Still life with fruits in oil painting style",
            "Fantasy castle on a floating island, digital art",
            "Minimalist geometric pattern",
            "Surreal dreamscape with floating objects",
            "Anime character with colorful background"
        ]
    
    # Repeat prompts if needed
    while len(prompts) < num_images:
        prompts.extend(prompts)
    prompts = prompts[:num_images]
    
    print(f"Loading SDXL model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    print(f"Generating {num_images} test images...")
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        # Generate image
        image = pipeline(prompt, num_inference_steps=30).images[0]
        
        # Save image
        image_path = os.path.join(output_dir, f"test_image_{i:03d}.png")
        image.save(image_path)
    
    print(f"Generated {num_images} test images in {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Test aesthetic reward model")
    parser.add_argument("--reward_model_path", type=str, default="output/final_model.pth",
                        help="Path to the aesthetic reward model weights")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory containing test images (if not provided, will generate test images)")
    parser.add_argument("--generate_images", action="store_true",
                        help="Generate test images using SDXL")
    parser.add_argument("--num_test_images", type=int, default=20,
                        help="Number of test images to generate (if --generate_images is set)")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="File containing prompts for image generation (one per line)")
    
    args = parser.parse_args()
    
    # Load reward model
    print(f"Loading aesthetic reward model from {args.reward_model_path}")
    reward_model = AestheticRewardModel(model_path=args.reward_model_path)
    
    # Plot reward function
    plot_reward_function(reward_model)
    
    # Generate or use existing test images
    if args.generate_images or args.images_dir is None:
        prompts = None
        if args.prompts_file:
            with open(args.prompts_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        
        images_dir = generate_test_images(
            output_dir=args.images_dir or "test_images",
            num_images=args.num_test_images,
            prompts=prompts
        )
    else:
        images_dir = args.images_dir
    
    # Test reward model on images
    test_reward_model(reward_model, images_dir)

if __name__ == "__main__":
    main()