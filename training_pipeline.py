import os
import argparse
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and print its output"""
    print(f"=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nCommand completed in {duration:.2f} seconds with return code: {process.returncode}")
    
    if process.returncode != 0:
        print(f"WARNING: Command exited with non-zero return code: {process.returncode}")
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run full SDXL RL training pipeline")
    parser.add_argument("--output_dir", type=str, default=f"outputs/sdxl_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Base output directory for all pipeline stages")
    parser.add_argument("--reward_model_path", type=str, default="output/final_model.pth",
                        help="Path to the aesthetic reward model")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base SDXL model to finetune")
    parser.add_argument("--num_prompts", type=int, default=1000,
                        help="Number of prompts to generate")
    parser.add_argument("--num_train_steps", type=int, default=1000,
                        help="Number of RL training steps")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="Weight for KL divergence loss")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate for training")
    parser.add_argument("--skip_prompt_generation", action="store_true",
                        help="Skip prompt generation step (use existing prompts file)")
    parser.add_argument("--skip_reward_testing", action="store_true",
                        help="Skip reward model testing step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the actual RL training step")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip final evaluation step")
    parser.add_argument("--test_prompts_file", type=str, default=None,
                        help="File with test prompts for evaluation (if not specified, will use a subset of training prompts)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="sdxl-rl",
                        help="W&B project name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths for artifacts
    prompts_file = os.path.join(args.output_dir, "prompts.txt")
    test_prompts_file = args.test_prompts_file or os.path.join(args.output_dir, "test_prompts.txt")
    test_images_dir = os.path.join(args.output_dir, "test_images")
    model_output_dir = os.path.join(args.output_dir, "model")
    eval_output_dir = os.path.join(args.output_dir, "evaluation")
    
    # 1. Generate prompts
    if not args.skip_prompt_generation:
        cmd = [
            "python", "prompt_generator.py",
            "--output_file", prompts_file,
            "--num_prompts", str(args.num_prompts)
        ]
        run_command(cmd, "Generating prompts")
        
        # If no test prompts file is provided, create one with a subset of training prompts
        if args.test_prompts_file is None:
            cmd = [
                "python", "-c", 
                f"import random; random.seed({args.seed}); "
                f"with open('{prompts_file}', 'r') as f: prompts = [line.strip() for line in f if line.strip()]; "
                f"test_prompts = random.sample(prompts, min(20, len(prompts))); "
                f"with open('{test_prompts_file}', 'w') as f: f.write('\\n'.join(test_prompts))"
            ]
            run_command(cmd, "Creating test prompts subset")
    
    # 2. Test reward model
    if not args.skip_reward_testing:
        cmd = [
            "python", "reward_testing.py",
            "--reward_model_path", args.reward_model_path,
            "--generate_images",
            "--images_dir", test_images_dir,
            "--num_test_images", "10",
            "--prompts_file", test_prompts_file
        ]
        run_command(cmd, "Testing reward model")
    
    # 3. Run RL training
    if not args.skip_training:
        cmd = [
            "python", "sdxl_rl_training.py",
            "--reward_model_path", args.reward_model_path,
            "--sd_model_path", args.base_model,
            "--prompt_file", prompts_file,
            "--output_dir", model_output_dir,
            "--learning_rate", str(args.learning_rate),
            "--num_steps", str(args.num_train_steps),
            "--kl_weight", str(args.kl_weight),
            "--seed", str(args.seed)
        ]
        
        if args.use_wandb:
            cmd.extend([
                "--use_wandb",
                "--wandb_project", args.wandb_project,
                "--wandb_run_name", f"sdxl-rl-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ])
        
        run_command(cmd, "Running RL training")
    
    # 4. Evaluate model
    if not args.skip_evaluation:
        # Final model path
        rl_model_path = os.path.join(model_output_dir, "checkpoints", "final")
        
        if not os.path.exists(rl_model_path):
            # Try to find the latest checkpoint
            checkpoints = [d for d in os.listdir(os.path.join(model_output_dir, "checkpoints")) 
                          if os.path.isdir(os.path.join(model_output_dir, "checkpoints", d))]
            
            if checkpoints:
                # Sort by step number if possible
                try:
                    checkpoints.sort(key=lambda x: int(x.split("_")[1]) if x.startswith("step_") else 0, reverse=True)
                except:
                    checkpoints.sort()
                
                rl_model_path = os.path.join(model_output_dir, "checkpoints", checkpoints[0])
                print(f"Final model not found, using latest checkpoint: {rl_model_path}")
            else:
                print("No checkpoints found for evaluation")
                return
        
        cmd = [
            "python", "comparison_script.py",
            "--base_model_path", args.base_model,
            "--rl_model_path", rl_model_path,
            "--reward_model_path", args.reward_model_path,
            "--prompts_file", test_prompts_file,
            "--output_dir", eval_output_dir,
            "--num_prompts", "10",
            "--seed", str(args.seed)
        ]
        run_command(cmd, "Evaluating models")
    
    print("\n=== Pipeline Completed Successfully ===")
    print(f"All outputs are saved in: {args.output_dir}")
    
    if not args.skip_evaluation:
        print(f"Check evaluation results in: {eval_output_dir}")
        print(f"Comparison report: {os.path.join(eval_output_dir, 'comparison_report.txt')}")

if __name__ == "__main__":
    main()