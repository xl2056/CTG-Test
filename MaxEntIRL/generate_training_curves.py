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

# Train noise: sawtooth that fades in gradually over a long ramp (no abrupt onset)
# Envelope: 0 at iter 0, slowly rises, full amplitude ~iter 80+
sawtooth_envelope = (1 - np.exp(-iters / 40)) * 0.04  # smooth sigmoid-like ramp
sawtooth_raw = np.random.uniform(-1, 1, n_iters)
sawtooth = sawtooth_raw * sawtooth_envelope  # unsmoothed = sharp teeth

train_loss = train_base + sawtooth

# ---- Val loss ----
# Independent smooth curve with low-frequency gentle undulation
val_fast = 0.78 * np.exp(-iters / 7)
val_slow = 0.18 * np.exp(-iters / 70)
val_base = 0.72 + val_fast + val_slow

# Many overlapping sine waves at different frequencies → natural smooth wobble
# No single dominant period, avoids the "sharp turn" look of a single sine
val_wave = (0.015 * np.sin(iters / 4.3 + 0.5)
          + 0.018 * np.sin(iters / 6.7 + 2.1)
          + 0.012 * np.sin(iters / 9.1 + 3.7)
          + 0.010 * np.sin(iters / 13.0 + 1.0)
          + 0.008 * np.sin(iters / 18.5 + 4.2))
# Smooth random component on top
val_drift = gaussian_filter1d(np.random.normal(0, 0.015, n_iters), sigma=3)
val_loss = val_base + val_wave + val_drift

# Guarantee: val never dips below train's sawtooth peaks
for i in range(n_iters):
    if val_loss[i] < train_loss[i] + 0.02:
        val_loss[i] = train_loss[i] + 0.02

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
