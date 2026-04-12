"""Generate simulated training curves for demonstration purposes."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

n_iters = 200
iters = np.arange(1, n_iters + 1)

# ---- Train loss: smooth exponential decay with slight noise ----
# Starts near ln(5)=1.61 (random baseline for 5 candidates), converges ~0.75
train_base = 0.75 + 0.86 * np.exp(-iters / 55)
train_noise = np.random.normal(0, 0.008, n_iters)
# Smooth the noise
from scipy.ndimage import gaussian_filter1d
train_noise = gaussian_filter1d(train_noise, sigma=2)
train_loss = train_base + train_noise

# ---- Val loss: follows train closely, slightly higher, converges ~0.82 ----
val_base = 0.82 + 0.78 * np.exp(-iters / 60)
val_noise = np.random.normal(0, 0.015, n_iters)
val_noise = gaussian_filter1d(val_noise, sigma=2)
val_loss = val_base + val_noise

# Ensure val >= train after initial crossover settles
for i in range(20, n_iters):
    if val_loss[i] < train_loss[i]:
        val_loss[i] = train_loss[i] + abs(np.random.normal(0.01, 0.005))

# ---- Log-likelihood: mirrors loss but inverted (higher = better) ----
ll_base = -1.1 + 0.5 * (1 - np.exp(-iters / 60))
ll_noise = gaussian_filter1d(np.random.normal(0, 0.01, n_iters), sigma=2)
log_likelihood = ll_base + ll_noise

# ---- Human likeness: decreasing (lower = more similar to expert) ----
hl_base = 3.2 - 1.4 * (1 - np.exp(-iters / 70))
hl_noise = gaussian_filter1d(np.random.normal(0, 0.03, n_iters), sigma=2)
human_likeness = hl_base + hl_noise

# ---- Weight norm: gradually increases then stabilizes ----
wn_base = 0.5 + 1.8 * (1 - np.exp(-iters / 40))
wn_noise = gaussian_filter1d(np.random.normal(0, 0.02, n_iters), sigma=2)
weight_norm = wn_base + wn_noise

# ---- Plot ----
panels = [
    ("Loss", [("train", train_loss, "tab:blue"), ("val", val_loss, "tab:orange")]),
    ("Avg Log-Likelihood", [("", log_likelihood, "tab:blue")]),
    ("Avg Human Likeness", [("", human_likeness, "tab:blue")]),
    ("Avg Weight Norm", [("", weight_norm, "tab:blue")]),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.5 * len(panels)))
for ax, (title, series) in zip(axes, panels):
    for label, vals, color in series:
        ax.plot(iters, vals, linewidth=1.2, color=color, label=label if label else None)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any(s[0] for s in series):
        ax.legend()

fig.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "irl_output")
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "training_curves.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"Saved to {save_path}")
