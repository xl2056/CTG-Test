"""Generate simulated weight evaluation plots for demonstration purposes."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

n_frames = 50
frames = np.arange(1, n_frames + 1)

FEATURE_NAMES = ["velocity", "a_long", "jerk", "a_lat", "f_thw", "l_thw", "r_thw"]
FEATURE_SHORT = ["vel", r"$a_{lon}$", "jerk", r"$a_{lat}$", r"$THW_f$", r"$THW_l$", r"$THW_r$"]

# ------------------------------------------------------------------ #
# Fixed baseline weights (same for all scenarios)                    #
# ------------------------------------------------------------------ #
baseline = np.array([0.05, 0.10, -0.15, 0.10, 0.40, 0.20, 0.15])

# ------------------------------------------------------------------ #
# Weight network outputs per scenario (base + smooth fluctuations)   #
# ------------------------------------------------------------------ #

def make_weights(bases, amplitudes, seed=0):
    """Generate smooth weight curves around base values."""
    rng = np.random.RandomState(seed)
    weights = np.zeros((n_frames, len(bases)))
    for i, (b, amp) in enumerate(zip(bases, amplitudes)):
        noise = gaussian_filter1d(rng.normal(0, 1, n_frames), sigma=3) * amp
        wave = amp * 0.6 * np.sin(frames / (5 + i * 1.3) + rng.uniform(0, 6))
        weights[:, i] = b + noise + wave
    return weights


# Highway Car-Following:
#   High velocity weight, front_thw critical, jerk penalized, low lateral
highway_base = [0.35, 0.15, -0.25, 0.05, 0.80, 0.15, 0.10]
highway_amp =  [0.04, 0.03, 0.02,  0.02, 0.05, 0.02, 0.02]
w_highway = make_weights(highway_base, highway_amp, seed=10)

# Intersection:
#   Negative velocity (slow down), high a_lateral (turning), high side THWs
intersection_base = [-0.20, 0.20, -0.20, 0.55, 0.45, 0.50, 0.45]
intersection_amp =  [0.05,  0.04, 0.03,  0.05, 0.04, 0.05, 0.04]
w_intersection = make_weights(intersection_base, intersection_amp, seed=20)

# Dense Traffic:
#   Negative velocity, front_thw very high, all THWs elevated, jerk penalized
dense_base = [-0.15, 0.05, -0.30, 0.20, 0.90, 0.45, 0.40]
dense_amp =  [0.04,  0.03, 0.03,  0.03, 0.06, 0.04, 0.04]
w_dense = make_weights(dense_base, dense_amp, seed=30)

# Sparse Road:
#   High velocity, THWs near zero (no neighbors), everything else relaxed
sparse_base = [0.45, 0.05, -0.10, 0.02, 0.10, 0.02, 0.02]
sparse_amp =  [0.03, 0.02, 0.02,  0.01, 0.02, 0.01, 0.01]
w_sparse = make_weights(sparse_base, sparse_amp, seed=40)

# ------------------------------------------------------------------ #
# Plot                                                               #
# ------------------------------------------------------------------ #

scenarios = [
    ("Highway Car-Following", w_highway),
    ("Intersection", w_intersection),
    ("Dense Traffic", w_dense),
    ("Sparse Road", w_sparse),
]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

for idx, (title, w_net) in enumerate(scenarios):
    ax = axes[idx]
    for fi in range(len(FEATURE_NAMES)):
        ax.plot(frames, w_net[:, fi], linewidth=1.3, color=colors[fi],
                label=FEATURE_SHORT[fi])
        ax.axhline(y=baseline[fi], linewidth=1.0, color=colors[fi],
                   linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="black", linewidth=0.4, alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
baseline_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
handles.append(baseline_line)
labels.append("Baseline (fixed)")
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 1.03))

fig.tight_layout(rect=[0, 0, 1, 0.95])
out_dir = os.path.join(os.path.dirname(__file__), "irl_output")
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "weight_evaluation.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {save_path}")
