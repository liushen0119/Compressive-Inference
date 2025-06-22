import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusion3Pipeline

# Set global random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Experiment configuration
compression_percentages = [i / 10 for i in range(1, 10)]  # Compression from 10% to 90%
compress_steps = list(range(4, 27, 2))  # Apply compression at steps 4, 6, ..., 26
num_trials = 50
prompt = "A cat sitting on a sofa."
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "results_1")
os.makedirs(output_dir, exist_ok=True)

# Load SD3 pipeline
model_path = "/root/autodl-tmp/sd3_medium"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

# Generate baseline images
baseline_images = []
for i in range(num_trials):
    out = pipe(prompt, guidance_scale=7.5, num_inference_steps=28).images[0]
    baseline_images.append(out)

# Save baseline images
baseline_dir = os.path.join(output_dir, "baseline")
os.makedirs(baseline_dir, exist_ok=True)
for i, img in enumerate(baseline_images):
    img.save(os.path.join(baseline_dir, f"baseline_{i}.png"))

# Run compression experiments
results = []
detailed_results = []

for step in compress_steps:
    for percentage in compression_percentages:
        psnr_vals, ssim_vals = [], []
        comp_dir = os.path.join(output_dir, f"compressed_step{step}_c{int(percentage*100)}")
        os.makedirs(comp_dir, exist_ok=True)
        for i in range(num_trials):
            out = pipe(prompt, compress_at_step=step, Compression_percentage=percentage).images[0]
            out.save(os.path.join(comp_dir, f"img_{i}.png"))
            baseline = baseline_images[i]
            img_arr = np.array(out).astype(np.float32) / 255.0
            ref_arr = np.array(baseline).astype(np.float32) / 255.0
            psnr_val = psnr(ref_arr, img_arr, data_range=1.0)
            ssim_val = ssim(ref_arr, img_arr, channel_axis=-1, data_range=1.0)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)
            detailed_results.append((step, percentage, i, psnr_val, ssim_val))
        results.append((step, percentage, np.mean(psnr_vals), np.std(psnr_vals), np.mean(ssim_vals), np.std(ssim_vals)))
        print(f"Step={step}, Compression={percentage:.1f} -> PSNR={np.mean(psnr_vals):.2f}, SSIM={np.mean(ssim_vals):.3f}")

# Save results
df = pd.DataFrame(results, columns=["step", "compression", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"])
df.to_csv(os.path.join(output_dir, "quality_metrics.csv"), index=False)

df_detailed = pd.DataFrame(detailed_results, columns=["step", "compression", "trial", "psnr", "ssim"])
df_detailed.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
print("Saved quality metrics and detailed scores to CSV files.")

# Plotting
plt.figure()
for step in compress_steps:
    subset = df[df.step == step]
    plt.plot(subset.compression, subset.psnr_mean, label=f"Step {step}")
plt.xlabel("Compression Percentage")
plt.ylabel("PSNR")
plt.legend()
plt.savefig(os.path.join(output_dir, "psnr_vs_compression_rate.png"))

plt.figure()
for step in compress_steps:
    subset = df[df.step == step]
    plt.plot(subset.compression, subset.ssim_mean, label=f"Step {step}")
plt.xlabel("Compression Percentage")
plt.ylabel("SSIM")
plt.legend()
plt.savefig(os.path.join(output_dir, "ssim_vs_compression_rate.png"))

plt.figure()
for pct in compression_percentages:
    subset = df[df.compression == pct]
    plt.plot(subset.step, subset.psnr_mean, label=f"{int(pct*100)}%")
plt.xlabel("Compression Step")
plt.ylabel("PSNR")
plt.legend()
plt.savefig(os.path.join(output_dir, "psnr_vs_step.png"))

plt.figure()
for pct in compression_percentages:
    subset = df[df.compression == pct]
    plt.plot(subset.step, subset.ssim_mean, label=f"{int(pct*100)}%")
plt.xlabel("Compression Step")
plt.ylabel("SSIM")
plt.legend()
plt.savefig(os.path.join(output_dir, "ssim_vs_step.png"))

print("Experiment completed. All results saved to 'results_1' folder.")
