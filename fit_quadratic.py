import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
CSV_IN = "./Experiment_results_1/compression_data_1.csv"     # Path to input CSV
OUT_DIR = "./Experiment_results_1/quadratic_fit"             # Output directory
BITS_TO_FIT = [8, 6, 4]                                      # Which bit-depths to fit, e.g. [8] or [4, 6]
INCLUDE_SCATTER = True                                       # Whether to plot raw scatter points
MIN_POINTS = 3                                               # Minimum distinct x values (steps) to perform quadratic fit

os.makedirs(OUT_DIR, exist_ok=True)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to unified schema: rate, step, ssim."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Map possible names
    col_map = {}
    col_map["rate"] = "compression_rate" if "compression_rate" in df.columns else ("rate" if "rate" in df.columns else None)
    col_map["step"] = "step" if "step" in df.columns else ("compress_at_step" if "compress_at_step" in df.columns else None)
    col_map["ssim"] = "ssim" if "ssim" in df.columns else ("ssim_gray" if "ssim_gray" in df.columns else ("ssim_value" if "ssim_value" in df.columns else None))

    if any(col_map[k] is None for k in ["rate","step","ssim"]):
        raise ValueError("CSV must contain rate/compression_rate, step/compress_at_step, ssim/ssim_gray/ssim_value")

    df = df.rename(columns={v:k for k,v in col_map.items()}).copy()
    for c in ["rate","step","ssim"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["rate","step","ssim"])

    # Derive bits: round((1 - rate) * 16), clipped to [1, 16]
    bits = np.round((1 - df["rate"]) * 16).astype(int)
    df["bits"] = np.clip(bits, 1, 16)
    return df

def quadfit(x,y):
    """Fit y = a*x^2 + b*x + c, return coefficients and R^2."""
    coefs = np.polyfit(x,y,2)
    p = np.poly1d(coefs)
    yhat = p(x)
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return coefs, r2

# Load and process data
df_raw = pd.read_csv(CSV_IN)
df = normalize_columns(df_raw)

# Aggregate mean SSIM across (bits, rate, step)
agg = df.groupby(["bits","rate","step"])["ssim"].mean().reset_index()

fits = []
fn_lines = []

plt.figure(figsize=(12,8))

for b in BITS_TO_FIT:
    sub_b = agg[agg["bits"]==b]
    for r in sorted(sub_b["rate"].unique()):
        sub = sub_b[sub_b["rate"]==r]
        if sub["step"].nunique() < MIN_POINTS:
            continue
        x = sub["step"].values
        y = sub["ssim"].values
        coefs, r2 = quadfit(x,y)
        a,bcoef,c = coefs.tolist()
        fits.append({"bits":b,"rate":r,"a":a,"b":bcoef,"c":c,
                     "r2":r2,"x_min":min(x),"x_max":max(x),"n":len(x)})
        fn_line = f"bits={b}, rate={r:.3f}: SSIM(x) = {a:.6f}*x^2 + {bcoef:.6f}*x + {c:.6f} (R^2={r2:.3f})"
        print(fn_line)  # Print function
        fn_lines.append(fn_line)

        # Plot fitted curve and scatter points
        xs = np.linspace(min(x),max(x),200)
        ys = a*xs**2 + bcoef*xs + c
        plt.plot(xs, ys, label=f"bits={b}, rate={r:.3f} fit")
        if INCLUDE_SCATTER:
            plt.scatter(x,y,s=18,alpha=0.6)

plt.xlabel("Compression Step (x)")
plt.ylabel("SSIM")
plt.title(f"Quadratic Fits (bits={BITS_TO_FIT})")
plt.grid(True,linestyle="--",alpha=0.5)
plt.legend(fontsize=8,ncol=2)
plt.tight_layout()

# Save figure
plot_path = os.path.join(OUT_DIR,"quadratic_fits_allbits.png")
plt.savefig(plot_path,dpi=150)
plt.close()

# Save coefficients and functions
pd.DataFrame(fits).to_csv(os.path.join(OUT_DIR,"quadratic_fit_coeffs.csv"),index=False)
with open(os.path.join(OUT_DIR,"quadratic_fit_functions.txt"),"w") as f:
    f.write("\n".join(fn_lines))

print("\n=== Saved files ===")
print("Figure:", plot_path)
print("Coefficients CSV:", os.path.join(OUT_DIR,'quadratic_fit_coeffs.csv'))
print("Functions TXT:", os.path.join(OUT_DIR,'quadratic_fit_functions.txt'))
