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
# Phase 1 (0-25): steep near-vertical drop from ~1.61 to ~0.75
# Phase 2 (25-200): slow convergence to ~0.54
# Use a two-stage exponential: fast + slow
train_fast = 0.85 * np.exp(-iters / 6)    # steep drop, dominant in first 25 iters
train_slow = 0.22 * np.exp(-iters / 80)   # gradual tail
train_base = 0.54 + train_fast + train_slow

# Noise: near-zero during steep phase, gradually increasing in plateau
noise_envelope = np.clip((iters - 20) / 60, 0, 1) * 0.012  # ramps from 0 to 0.012
train_noise = np.random.normal(0, 1, n_iters) * noise_envelope
train_noise = gaussian_filter1d(train_noise, sigma=1.5)
train_loss = train_base + train_noise

# ---- Val loss ----
# Slightly different start point (a bit lower than train at iter 1)
# Never crosses train — always above after the first few iters
val_fast = 0.78 * np.exp(-iters / 7)
val_slow = 0.18 * np.exp(-iters / 70)
val_base = 0.65 + val_fast + val_slow

# Val noise: also ramps up, slightly larger than train
val_noise_envelope = np.clip((iters - 20) / 50, 0, 1) * 0.018
val_noise = np.random.normal(0, 1, n_iters) * val_noise_envelope
val_noise = gaussian_filter1d(val_noise, sigma=1.5)
val_loss = val_base + val_noise

# Guarantee: val always >= train (no crossing), with a small natural gap
for i in range(n_iters):
    min_gap = 0.03 + 0.08 * (1 - np.exp(-i / 30))  # gap grows from 0.03 to ~0.11
    if val_loss[i] < train_loss[i] + min_gap:
        val_loss[i] = train_loss[i] + min_gap + abs(np.random.normal(0, 0.003))

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
