import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

# ===== 1. Load data =====
df = pd.read_csv("./Experiment_results_2/compression_all_data.csv")

# ===== 2. Normalize PSNR and SSIM, then compute quality score =====
psnr = df["PSNR"].values.reshape(-1, 1)
ssim = df["SSIM"].values.reshape(-1, 1)

scaler_psnr = MinMaxScaler()
scaler_ssim = MinMaxScaler()
psnr_norm = scaler_psnr.fit_transform(psnr).flatten()
ssim_norm = scaler_ssim.fit_transform(ssim).flatten()

score = 0.5 * psnr_norm + 0.5 * ssim_norm
quality = np.log(score + 1e-6)  # log mapping to smooth range

# ===== 3. Normalize input variables: step and ratio =====
s_raw = df["compress_at_step"].values.reshape(-1, 1)
r_raw = df["compression_ratio"].values.reshape(-1, 1)

scaler_s = MinMaxScaler()
scaler_r = MinMaxScaler()
s_norm = scaler_s.fit_transform(s_raw).flatten()
r_norm = scaler_r.fit_transform(r_raw).flatten()

X_data = (s_norm, r_norm)

# ===== 4. Define 2D sigmoid function =====
def sigmoid2(X, L, K1, B1, K2, B2, D):
    s, r = X
    return L / (1 + np.exp(K1 * (s - B1) + K2 * (r - B2))) + D

# ===== 5. Fit the model with bounds =====
lower_bounds = [0.1, -50, 0.0, -50, 0.0, -5.0]
upper_bounds = [2.0,  50, 1.0,  50, 1.0,  5.0]

popt, _ = curve_fit(
    sigmoid2, X_data, quality,
    p0=[1, -10, 0.5, 10, 0.5, 0],
    bounds=(lower_bounds, upper_bounds)
)
L, K1, B1, K2, B2, D = popt

# ===== 6. Print fitted normalized sigmoid formula =====
print("\nFitted sigmoid function in normalized space:")
print(f"Q(s', r') = {L:.4f} / (1 + exp({K1:.4f} * (s' - {B1:.4f}) + {K2:.4f} * (r' - {B2:.4f}))) + {D:.4f}")
print("Note: s' and r' are normalized to [0, 1]")

# Also save to a text file
with open("./Experiment_results_2/fitted_sigmoid_formula.txt", "w") as f:
    f.write(f"Q(s', r') = {L:.6f} / (1 + exp({K1:.6f} * (s' - {B1:.6f}) + {K2:.6f} * (r' - {B2:.6f}))) + {D:.6f}\n")

# ===== 7. Visualize the fitted surface =====
s_vals = np.linspace(0, 1, 50)
r_vals = np.linspace(0, 1, 50)
s_grid, r_grid = np.meshgrid(s_vals, r_vals)
q_grid = sigmoid2((s_grid, r_grid), *popt)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(s_grid, r_grid, q_grid, cmap="viridis")
ax.set_xlabel("Normalized Step (s')")
ax.set_ylabel("Normalized Ratio (r')")
ax.set_zlabel("Predicted Quality")
plt.title("Fitted 2D Sigmoid Surface")
plt.tight_layout()
plt.savefig("./Experiment_results_2/sigmoid_fit_surface_normalized.png")
plt.close()
print("Plot saved to sigmoid_fit_surface_normalized.png")
