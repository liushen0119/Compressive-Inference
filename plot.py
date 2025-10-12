import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "./Experiment_results_1"        # Root folder containing CSV
CSV_FILENAME = "compression_data_1.csv"      # CSV file name inside OUTPUT_DIR
PLOT_SUBDIR = "plots"                        # Subfolder for saving plots
BITS_TO_PLOT = list(range(1, 17))            # Which bits to include (e.g., [8], [4,6], [8,6,4])                 
SHOW_ERROR_BARS = False                      # Whether to show ±std error bars

# Build paths
csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
plot_dir = os.path.join(OUTPUT_DIR, PLOT_SUBDIR)
os.makedirs(plot_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# Normalize columns: expect fields for 'rate', 'step', 'ssim'
cols = {c.lower(): c for c in df.columns}
def pick(*names):
    for n in names:
        if n in cols:
            return cols[n]
    return None

rate_col = pick("compression_rate", "rate")
step_col = pick("step", "compress_at_step")
ssim_col = pick("ssim", "ssim_gray", "ssim_value")
if not (rate_col and step_col and ssim_col):
    raise ValueError("CSV must contain columns for rate, step, and ssim (or compatible aliases).")

df = df[[step_col, rate_col, ssim_col]].copy()
df.columns = ["step", "rate", "ssim"]
df["step"] = pd.to_numeric(df["step"], errors="coerce")
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
df = df.dropna(subset=["step","rate","ssim"])

# Map rate -> bits using formula: bits = round((1 - rate) * 16), clipped to [1, 16]
bits = np.round((1.0 - df["rate"]) * 16.0).astype(int)
bits = np.clip(bits, 1, 16)
df["bits"] = bits

# Filter by chosen bits
df = df[df["bits"].isin(BITS_TO_PLOT)]
if df.empty:
    raise ValueError(f"No data matched the chosen bits {BITS_TO_PLOT}")

# Group by (step, rate)
grouped = df.groupby(["step", "rate"]).agg({"ssim": ["mean", "std"]}).reset_index()
grouped.columns = ["step", "rate", "ssim_mean", "ssim_std"]

# Plot 1: SSIM vs Compression Rate
def plot_vs_rate(df, filename):
    plt.figure(figsize=(9, 6))
    for step in sorted(df["step"].unique()):
        subset = df[df["step"] == step].sort_values("rate", ascending=False)
        x = subset["rate"].values
        y = subset["ssim_mean"].values
        if SHOW_ERROR_BARS:
            yerr = subset["ssim_std"].values
            plt.errorbar(x, y, yerr=yerr, label=f"Step {step}", marker="o", capsize=3)
        else:
            plt.plot(x, y, label=f"Step {step}", marker="o")

    plt.xlabel("Compression Rate")
    plt.ylabel("SSIM")
    plt.title(f"SSIM vs Compression Rate (bits={BITS_TO_PLOT})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

# Plot 2: SSIM vs Step
def plot_vs_step(df, filename):
    plt.figure(figsize=(9, 6))
    for rate in sorted(df["rate"].unique(), reverse=True):
        subset = df[df["rate"] == rate].sort_values("step")
        x = subset["step"].values
        y = subset["ssim_mean"].values
        if SHOW_ERROR_BARS:
            yerr = subset["ssim_std"].values
            plt.errorbar(x, y, yerr=yerr, label=f"Rate {rate:.3f}", marker="s", capsize=3)
        else:
            plt.plot(x, y, label=f"Rate {rate:.3f}", marker="s")

    plt.xlabel("Compress At Step")
    plt.ylabel("SSIM")
    plt.title(f"SSIM vs Compress Step (bits={BITS_TO_PLOT})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

# Execute
plot_vs_rate(grouped, "ssim_vs_rate1.png")
plot_vs_step(grouped, "ssim_vs_step1.png")
print(f"Plots saved to {plot_dir}/")
