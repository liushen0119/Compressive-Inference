import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from skimage.metrics import structural_similarity as ssim

# Configuration
model_path = "../sd3_medium"
output_dir = "./Experiment_results_1"
image_dir = os.path.join(output_dir, "images")
os.makedirs(image_dir, exist_ok=True)

prompts = [
    "A domestic cat sitting on a wooden floor",
    "A golden retriever standing on grass",
    "A sleeping cat lying on a white background",
    "A close-up of a Ragdoll cat's face against a gray backdrop",
    "Two puppies playing on an indoor floor",
    "A Border Collie running with a blurred background",
    "A black-and-white portrait of a Persian cat showing fur texture",
    "A cat resting on a couch",
    "An orange tabby cat sitting in an outdoor yard",
    "A Labrador Retriever walking down a street"
]

total_steps = 28
compress_steps = [2,4,6,8,10,12,14,16,18,20,22,24,26]
compression_rate = [0.0,0.25,0.375,0.5,0.625,0.6875,0.75,0.8125,0.875]
repeats = 5                   
guidance_scale = 7.0
seed_base = 42

# Image conversion
def to_gray_np(img):
    return np.array(img.convert("L")) / 255.0

# Load Stable Diffusion 3 model
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")

# Generate original images with progress
original_images = {}
original_np_maps = {}
print("Generating original images...")
for p_idx, prompt in enumerate(prompts):
    for i in tqdm(range(repeats), desc=f"Orig gen P{p_idx}", unit="img"):
        seed = seed_base + i
        print(f"  -> Generating original image for prompt {p_idx}, seed {seed}...")
        generator = torch.Generator("cuda").manual_seed(seed)
        img = pipe(
            prompt=prompt,
            num_inference_steps=total_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        original_images[(p_idx, seed)] = img
        original_np_maps[(p_idx, seed)] = to_gray_np(img)
        path = os.path.join(image_dir, f"orig_p{p_idx}_seed{seed}.png")
        print(f"     Saving original image to {path}")
        img.save(path)

# Compression experiments
results = []
print("Starting compression experiments...")
for p_idx, prompt in enumerate(prompts):
    for step in tqdm(compress_steps, desc=f"Compress P{p_idx}", unit="step"):
        for rate in compression_rate:
            for i in range(repeats):
                seed = seed_base + i
                print(f"  -> Generating compressed image for prompt {p_idx}, step={step}, rate={rate}, seed={seed}...")
                generator = torch.Generator("cuda").manual_seed(seed)
                img_compressed = pipe(
                    prompt=prompt,
                    num_inference_steps=total_steps,
                    guidance_scale=guidance_scale,
                    compress_at_step=step,
                    compression_rate=rate,
                    generator=generator
                ).images[0]
                # Compute SSIM
                original_np = original_np_maps[(p_idx, seed)]
                compressed_np = to_gray_np(img_compressed)
                ssim_val = ssim(original_np, compressed_np, data_range=1.0)
                print(f"     Computed SSIM: {ssim_val:.4f}")
                # Save image
                name_prefix = f"p{p_idx}_step{step}_rate{int(rate*100):03d}_seed{seed}"
                path = os.path.join(image_dir, f"{name_prefix}_compressed.png")
                print(f"     Saving compressed image to {path}")
                img_compressed.save(path)
                # Record
                results.append({
                    "prompt_index": p_idx,
                    "prompt_text": prompt,
                    "step": step,
                    "rate": round(rate, 4),
                    "seed": seed,
                    "ssim": round(ssim_val, 4),
                })

# Save to CSV
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "compression_data_1.csv")
df.to_csv(csv_path, index=False)
print(f"Saved results to: {csv_path}")
