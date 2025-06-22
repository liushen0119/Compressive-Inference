import os 
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ========= Parameter Configuration =========
model_path = "../sd3_medium"
output_dir = "./Experiment_results_2"
image_dir = os.path.join(output_dir, "images")
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

prompt = "A futuristic robot standing in a neon-lit alley"
total_steps = 28
compress_steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
compression_ratios = [1.0, 0.9, 0.8, 0.75, 0.66, 0.5, 0.4, 0.33, 0.25]
repeats = 50
guidance_scale = 7.0
seed_base = 42

def to_gray_np(img):
    return np.array(img.convert("L")) / 255.0

# ========= Load Model =========
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")

# ========= Generate Original Images =========
print("\nGenerating original images...")
original_images = {}
original_np_maps = {}
for i in range(repeats):
    seed = seed_base + i
    generator = torch.Generator("cuda").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        num_inference_steps=total_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    original_images[seed] = img
    original_np_maps[seed] = to_gray_np(img)
    img.save(os.path.join(image_dir, f"original_seed{seed}.png"))
print(f"Original images generated: {repeats} in total.")

# ========= Compression Experiment Loop =========
results = []
print("\nStarting compression experiments...")
for step in tqdm(compress_steps, desc="Sweep compress_at_step"):
    for ratio in compression_ratios:
        scale = round(1.0 / ratio, 2)
        for i in range(repeats):
            seed = seed_base + i
            generator = torch.Generator("cuda").manual_seed(seed)

            image_original = original_images[seed]
            original_np = original_np_maps[seed]

            image_compressed = pipe(
                prompt=prompt,
                num_inference_steps=total_steps,
                guidance_scale=guidance_scale,
                compress_at_step=step,
                compress_scale=scale,
                generator=generator
            ).images[0]
            compressed_np = to_gray_np(image_compressed)

            psnr_val = psnr(original_np, compressed_np, data_range=1.0)
            ssim_val = ssim(original_np, compressed_np, data_range=1.0)

            name_prefix = f"step{step}_ratio{ratio}_seed{seed}"
            image_compressed.save(os.path.join(image_dir, f"{name_prefix}_compressed.png"))

            results.append({
                "compress_at_step": step,
                "compression_ratio": round(ratio, 4),
                "compress_scale": scale,
                "seed": seed,
                "original_path": f"original_seed{seed}.png",
                "compressed_path": f"{name_prefix}_compressed.png",
                "PSNR": round(psnr_val, 2),
                "SSIM": round(ssim_val, 4),
            })

# ========= Save CSV =========
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "compression_all_data.csv")
df.to_csv(csv_path, index=False)
print(f"\nAll data saved to: {csv_path}")

# ========= Plotting =========
print("\nGenerating plots...")
grouped = df.groupby(["compress_at_step", "compression_ratio"]).agg({
    "PSNR": ["mean", "std"],
    "SSIM": ["mean", "std"]
}).reset_index()
grouped.columns = ["step", "ratio", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]

def plot_trend(df, metric="ssim_mean", ylabel="SSIM", save_name="ssim_vs_ratio.png"):
    plt.figure(figsize=(9, 6))
    for step in sorted(df["step"].unique()):
        subset = df[df["step"] == step].sort_values("ratio", ascending=False)
        plt.plot(subset["ratio"], subset[metric], label=f"Step {step}", marker="o")

    plt.xlabel("Compression Ratio")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Compression Ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name))
    plt.close()

def plot_by_step(df, metric="ssim_mean", ylabel="SSIM", save_name="ssim_vs_step.png"):
    plt.figure(figsize=(9, 6))
    for ratio in sorted(df["ratio"].unique(), reverse=True):
        subset = df[df["ratio"] == ratio].sort_values("step")
        plt.plot(subset["step"], subset[metric], label=f"Ratio {ratio}", marker="s")

    plt.xlabel("Compress At Step")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Compress Step")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name))
    plt.close()

plot_trend(grouped, metric="ssim_mean", ylabel="SSIM", save_name="ssim_vs_ratio.png")
plot_trend(grouped, metric="psnr_mean", ylabel="PSNR", save_name="psnr_vs_ratio.png")
plot_by_step(grouped, metric="ssim_mean", ylabel="SSIM", save_name="ssim_vs_step.png")
plot_by_step(grouped, metric="psnr_mean", ylabel="PSNR", save_name="psnr_vs_step.png")
print("All plots saved in:", plot_dir)
