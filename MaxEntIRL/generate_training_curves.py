"""Generate simulated training curves for demonstration purposes."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

n_iters = 200
iters = np.arange(1, n_iters + 1)

# ---- Train loss ----
# Two-stage staircase descent: steep drop → shelf → second drop → smooth convergence
# Matches reference blue curve shape. Converges to ~0.30.
drop1 = 1.00 * np.exp(-iters / 5)                         # steep initial drop
drop2 = 0.25 * np.exp(-np.maximum(iters - 35, 0) / 15)    # delayed second drop
train_base = 0.30 + drop1 + drop2

# Bumps concentrated in transition zone (iter 50-130), quiet at start and end
bump_env = np.exp(-((iters - 80) / 30) ** 2) * 0.035
train_bumps = gaussian_filter1d(np.random.uniform(-1, 1, n_iters), sigma=0.8) * bump_env

train_loss = train_base + train_bumps

# ---- Val loss ----
# Same two-stage staircase shape, converges to ~0.40
val_rng = np.random.RandomState(seed=99)
val_drop1 = 0.90 * np.exp(-iters / 6)
val_drop2 = 0.20 * np.exp(-np.maximum(iters - 40, 0) / 18)
val_base = 0.40 + val_drop1 + val_drop2

# Smooth undulation (higher freq, smaller amplitude than before)
val_wave = (0.010 * np.sin(iters / 2.3 + 1.8)
          + 0.008 * np.sin(iters / 3.7 + 4.0)
          + 0.007 * np.sin(iters / 5.2 + 0.6)
          + 0.006 * np.sin(iters / 7.1 + 2.9)
          + 0.005 * np.sin(iters / 9.8 + 5.1)
          + 0.004 * np.sin(iters / 13.0 + 3.3))
val_drift = gaussian_filter1d(val_rng.normal(0, 0.010, n_iters), sigma=3)
val_loss = val_base + val_wave + val_drift

# Soft guarantee: val stays above train without copying train's shape
for i in range(n_iters):
    if val_loss[i] < train_base[i] + 0.06:
        val_loss[i] = train_base[i] + 0.06

# ---- Log-likelihood: mirrors loss (higher = better) ----
ll_base = -1.15 + 0.55 * (1 - np.exp(-iters / 50))
ll_noise_env = np.clip((iters - 20) / 60, 0, 1) * 0.012
ll_noise = gaussian_filter1d(np.random.normal(0, 1, n_iters) * ll_noise_env, sigma=1.5)
log_likelihood = ll_base + ll_noise

# ---- Human likeness: decreasing (lower = closer to expert) ----
hl_base = 3.2 - 1.5 * (1 - np.exp(-iters / 55))
hl_noise_env = np.clip((iters - 20) / 60, 0, 1) * 0.035
hl_noise = gaussian_filter1d(np.random.normal(0, 1, n_iters) * hl_noise_env, sigma=1.5)
human_likeness = hl_base + hl_noise

# ---- Weight norm: increases then stabilizes ----
wn_base = 0.5 + 1.8 * (1 - np.exp(-iters / 35))
wn_noise_env = np.clip((iters - 15) / 50, 0, 1) * 0.025
wn_noise = gaussian_filter1d(np.random.normal(0, 1, n_iters) * wn_noise_env, sigma=1.5)
weight_norm = wn_base + wn_noise

# ---- Plot ----
panels = [
    ("Loss", [("train", train_loss, "tab:blue"), ("val", val_loss, "tab:orange")]),
    ("Avg Log-Likelihood", [("", log_likelihood, "tab:blue")]),
    ("Avg Human Likeness", [("", human_likeness, "tab:blue")]),
    ("Avg Weight Norm", [("", weight_norm, "tab:blue")]),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(5, 3.5 * len(panels)))
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
