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
# Smooth continuous descent (no shelf/plateau), converges to ~0.30
train_base = 0.30 + 1.00 * np.exp(-iters / 8) + 0.08 * np.exp(-iters / 45)

# Micro noise so convergence isn't a perfect flat line
train_micro = gaussian_filter1d(np.random.normal(0, 0.004, n_iters), sigma=1)

# Isolated spikes (毛刺) - each has unique irregular shape, small amplitude
train_spikes = np.zeros(n_iters)
train_spikes[48] = 0.006
train_spikes[91] = 0.003; train_spikes[92] = 0.007; train_spikes[93] = 0.002
train_spikes[156] = 0.005; train_spikes[157] = 0.004

train_loss = train_base + train_micro + train_spikes

# ---- Val loss ----
# Same smooth descent shape, converges to ~0.40, no upward curl
val_rng = np.random.RandomState(seed=99)
val_base = 0.40 + 0.90 * np.exp(-iters / 9) + 0.06 * np.exp(-iters / 40)

# Micro noise
val_micro = gaussian_filter1d(val_rng.normal(0, 0.005, n_iters), sigma=1.5)

# Isolated spikes - different shapes from train
val_spikes = np.zeros(n_iters)
val_spikes[65] = 0.005; val_spikes[66] = 0.007
val_spikes[118] = 0.006

val_loss = val_base + val_micro + val_spikes

# Soft guarantee: val stays above train
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
