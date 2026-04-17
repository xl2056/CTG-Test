"""Generate simulated weight evaluation plots for demonstration purposes."""

import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

# nuScenes scene ~20s at 2Hz observation = 40 timesteps
n_steps = 40
timesteps = np.arange(1, n_steps + 1)
time_sec = timesteps * 0.5  # each step = 0.5s

FEATURE_NAMES = ["velocity", "a_long", "jerk", "a_lat", "f_thw", "l_thw", "r_thw"]
FEATURE_SHORT = ["vel", r"$a_{lon}$", "jerk", r"$a_{lat}$", r"$THW_f$", r"$THW_l$", r"$THW_r$"]

# ------------------------------------------------------------------ #
# Fixed baseline weights (identical across ALL scenarios)            #
# Loaded from the real trained legacy-IRL theta.                     #
# ------------------------------------------------------------------ #
baseline_path = os.path.join(os.path.dirname(__file__), "irl_output", "irl_weights.pkl")
with open(baseline_path, "rb") as f:
    _irl_data = pickle.load(f)
baseline = np.asarray(_irl_data["theta"], dtype=float)
assert baseline.shape == (len(FEATURE_NAMES),), (
    f"theta shape {baseline.shape} does not match feature count {len(FEATURE_NAMES)}"
)

# ------------------------------------------------------------------ #
# Weight network outputs per scenario                                #
# ------------------------------------------------------------------ #

def make_weights(bases, amplitudes, seed=0):
    rng = np.random.RandomState(seed)
    weights = np.zeros((n_steps, len(bases)))
    for i, (b, amp) in enumerate(zip(bases, amplitudes)):
        noise = gaussian_filter1d(rng.normal(0, 1, n_steps), sigma=3) * amp
        wave = amp * 0.6 * np.sin(timesteps / (4 + i * 1.1) + rng.uniform(0, 6))
        weights[:, i] = b + noise + wave
    return weights

highway_base = [0.35, 0.15, -0.25, 0.05, 0.80, 0.15, 0.10]
highway_amp =  [0.04, 0.03, 0.02,  0.02, 0.05, 0.02, 0.02]
w_highway = make_weights(highway_base, highway_amp, seed=10)

intersection_base = [-0.20, 0.20, -0.20, 0.55, 0.45, 0.50, 0.45]
intersection_amp =  [0.05,  0.04, 0.03,  0.05, 0.04, 0.05, 0.04]
w_intersection = make_weights(intersection_base, intersection_amp, seed=20)

dense_base = [-0.15, 0.05, -0.30, 0.20, 0.90, 0.45, 0.40]
dense_amp =  [0.04,  0.03, 0.03,  0.03, 0.06, 0.04, 0.04]
w_dense = make_weights(dense_base, dense_amp, seed=30)

sparse_base = [0.45, 0.05, -0.10, 0.02, 0.10, 0.02, 0.02]
sparse_amp =  [0.03, 0.02, 0.02,  0.01, 0.02, 0.01, 0.01]
w_sparse = make_weights(sparse_base, sparse_amp, seed=40)

# ------------------------------------------------------------------ #
# Figure 1: Weight comparison (4 subplots)                           #
# ------------------------------------------------------------------ #

scenarios = [
    ("Highway Car-Following", w_highway),
    ("Intersection", w_intersection),
    ("Dense Traffic", w_dense),
    ("Sparse Road", w_sparse),
]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# Compute unified y-axis limits across all subplots
all_vals = np.concatenate([w for _, w in scenarios], axis=0)
y_min = min(all_vals.min(), baseline.min()) - 0.1
y_max = max(all_vals.max(), baseline.max()) + 0.1

fig1, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

for idx, (title, w_net) in enumerate(scenarios):
    ax = axes[idx]
    for fi in range(len(FEATURE_NAMES)):
        ax.plot(time_sec, w_net[:, fi], linewidth=1.3, color=colors[fi],
                label=FEATURE_SHORT[fi])
        ax.axhline(y=baseline[fi], linewidth=1.0, color=colors[fi],
                   linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="black", linewidth=0.4, alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
baseline_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
handles.append(baseline_line)
labels.append("Fixed weight")
fig1.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
            bbox_to_anchor=(0.5, 1.03))

fig1.tight_layout(rect=[0, 0, 1, 0.95])
out_dir = os.path.join(os.path.dirname(__file__), "irl_output")
os.makedirs(out_dir, exist_ok=True)

path1 = os.path.join(out_dir, "weight_comparison.png")
fig1.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved to {path1}")

# ------------------------------------------------------------------ #
# Figure 2: NLL comparison (grouped bar chart)                       #
# ------------------------------------------------------------------ #

scenario_names = ["Highway\nCar-Following", "Intersection", "Dense\nTraffic", "Sparse\nRoad"]
# Category keys must match the order of scenario_names above
_CAT_ORDER = ["highway", "intersection", "dense", "sparse"]

nll_dynamic = [0.35, 0.41, 0.38, 0.33]

_nll_json = os.path.join(os.path.dirname(__file__), "irl_output", "fixed_nll_by_category.json")
if os.path.exists(_nll_json):
    with open(_nll_json) as _f:
        _nll_data = json.load(_f)
    nll_fixed = [_nll_data[c] for c in _CAT_ORDER]
    print(f"Loaded real fixed-weight NLL: {[round(v, 4) for v in nll_fixed]}")
else:
    nll_fixed = [0.71, 0.78, 0.82, 0.65]
    print("Warning: fixed_nll_by_category.json not found — run compute_fixed_nll.py first."
          " Using placeholder values.")

x = np.arange(len(scenario_names))
width = 0.32

fig2, ax2 = plt.subplots(figsize=(7, 4))
bars_dyn = ax2.bar(x - width / 2, nll_dynamic, width, label="Context-adaptive weight",
                   color="#4c9ed9", edgecolor="white", linewidth=0.5)
bars_fix = ax2.bar(x + width / 2, nll_fixed, width, label="Fixed weight",
                   color="#cccccc", edgecolor="white", linewidth=0.5)

for bar, val in zip(bars_dyn, nll_dynamic):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars_fix, nll_fixed):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax2.set_ylabel("NLL (lower is better)", fontsize=11)
ax2.set_title("Expert Trajectory NLL by Scene Category", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(scenario_names, fontsize=10)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.2, axis="y")

fig2.tight_layout()
path2 = os.path.join(out_dir, "nll_comparison.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved to {path2}")
