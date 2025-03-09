import os
import random
import argparse
from tqdm import tqdm

# Art styles
ART_STYLES = [
    "oil painting", "watercolor", "digital art", "illustration", "concept art",
    "realistic", "photorealistic", "anime", "manga", "sketch", "cartoon",
    "3D render", "cinematic", "isometric", "cyberpunk", "steampunk", "fantasy",
    "sci-fi", "surrealism", "impressionism", "expressionism", "abstract",
    "cubism", "pop art", "minimalist", "vaporwave", "pixel art", "ukiyo-e"
]

# Lighting effects
LIGHTING = [
    "natural lighting", "soft lighting", "ambient lighting", "dramatic lighting",
    "cinematic lighting", "rim lighting", "volumetric lighting", "subsurface scattering",
    "studio lighting", "neon lighting", "golden hour", "blue hour", "sunset lighting",
    "moonlight", "backlit", "atmospheric", "low key", "high key", "chiaroscuro"
]

# Quality markers
QUALITY = [
    "highly detailed", "intricate details", "masterpiece", "trending on artstation",
    "award winning", "stunning", "beautiful", "professional", "precise linework",
    "perfect composition", "octane render", "unreal engine", "4k", "8k", "hdr"
]

# Subjects
SUBJECTS = [
    # Nature
    "mountain landscape", "forest", "beach sunset", "rolling hills", "waterfall",
    "desert", "ocean waves", "winter landscape", "autumn forest", "starry night sky",
    
    # Urban
    "futuristic city", "ancient castle", "cyberpunk street", "cozy cafe",
    "abandoned building", "modern architecture", "fantasy castle", "small village",
    
    # Characters
    "portrait of a warrior", "fantasy character", "wizard", "elf", "robot",
    "cyborg", "astronaut", "pirate", "knight", "samurai",
    
    # Objects
    "crystal gemstone", "mechanical watch", "ancient artifact", "magical staff",
    "spacecraft", "robot companion", "fantasy weapon", "treasure chest",
    
    # Abstract
    "fractal patterns", "geometric shapes", "abstract landscape", "surreal dreamscape",
    "psychedelic vision", "cosmic energy", "flowing liquid colors"
]

# Camera angles and perspectives
PERSPECTIVES = [
    "wide angle", "close-up", "aerial view", "bird's eye view", "isometric view",
    "dutch angle", "low angle", "high angle", "panoramic", "macro shot",
    "telephoto", "fish-eye lens", "drone shot", "portrait orientation"
]

def generate_prompt():
    """Generate a random artistic prompt for SDXL"""
    
    # Structure prompts with weighted randomness
    elements = []
    
    # Subject (100% chance)
    elements.append(random.choice(SUBJECTS))
    
    # Art style (90% chance)
    if random.random() < 0.9:
        elements.append(random.choice(ART_STYLES))
    
    # Lighting (60% chance)
    if random.random() < 0.6:
        elements.append(random.choice(LIGHTING))
    
    # Perspective (40% chance)
    if random.random() < 0.4:
        elements.append(random.choice(PERSPECTIVES))
    
    # Quality markers (70% chance for first, then decreasing probability)
    quality_count = 0
    quality_options = QUALITY.copy()
    random.shuffle(quality_options)
    
    for quality in quality_options:
        if random.random() < (0.7 / (quality_count + 1)):
            elements.append(quality)
            quality_count += 1
            if quality_count >= this3:
                break
    
    # Randomize order a bit to create variation
    random.shuffle(elements)
    
    # Construct prompt
    prompt = ", ".join(elements)
    
    # Capitalize first letter
    prompt = prompt[0].upper() + prompt[1:]
    
    return prompt

def generate_prompt_file(output_file, num_prompts=1000):
    """Generate a file with random prompts for SDXL training"""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    prompts = []
    for _ in tqdm(range(num_prompts), desc="Generating prompts"):
        prompts.append(generate_prompt())
    
    # Ensure uniqueness
    prompts = list(set(prompts))
    print(f"Generated {len(prompts)} unique prompts")
    
    with open(output_file, "w") as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    print(f"Saved prompts to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for SDXL training")
    parser.add_argument("--output_file", type=str, default="prompts.txt",
                        help="Output file to save prompts")
    parser.add_argument("--num_prompts", type=int, default=1000,
                        help="Number of prompts to generate")
    
    args = parser.parse_args()
    
    generate_prompt_file(args.output_file, args.num_prompts)