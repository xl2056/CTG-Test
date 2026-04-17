"""Evaluate the context-conditional weight network across scene categories.

Classifies extracted scenes into 4 types (highway car-following, intersection,
sparse road, dense traffic), selects one representative per category, and
visualizes how the weight network adapts its 7-dim output across frames.
"""

import os
import pickle
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .weight_network import WeightNetwork


FEATURE_NAMES = [
    "velocity", "a_long", "jerk_long", "a_lateral",
    "front_thw", "left_thw", "right_thw",
]

CATEGORIES = ["intersection", "highway", "dense", "sparse"]
CATEGORY_LABELS = {
    "intersection": "Intersection",
    "highway": "Highway Car-Following",
    "dense": "Dense Traffic",
    "sparse": "Sparse Road",
}


# ------------------------------------------------------------------ #
# Scene statistics extraction                                        #
# ------------------------------------------------------------------ #

def compute_scene_stats(scene_data: List[dict]) -> Dict[str, float]:
    """Compute aggregate statistics from a scene's frame entries."""
    all_speeds = []
    front_thw_values = []
    neighbor_counts = []
    max_cumulative_yaw = 0.0

    for frame_entry in scene_data:
        ff = frame_entry["frame_features"]
        gt = ff["agent_ground_truth_features"]

        for agent_id, feat in gt.items():
            vel = np.asarray(feat["velocity"])
            if vel.size > 0:
                all_speeds.append(float(np.mean(vel)))
            fthw = np.asarray(feat["front_thw"])
            if fthw.size > 0:
                front_thw_values.extend(fthw.tolist())

        ctx = frame_entry.get("context")
        if ctx is not None:
            nbr_pos = ctx.get("all_other_agents_history_positions")
            if nbr_pos is not None:
                arr = np.asarray(nbr_pos)
                if arr.ndim >= 2:
                    neighbor_counts.append(arr.shape[-3] if arr.ndim >= 3 else arr.shape[0])

            yaws = ctx.get("history_yaws")
            if yaws is not None:
                yaw_arr = np.asarray(yaws).flatten()
                if yaw_arr.size >= 2:
                    cumulative = float(np.abs(yaw_arr[-1] - yaw_arr[0]))
                    max_cumulative_yaw = max(max_cumulative_yaw, cumulative)

    mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
    front_thw_arr = np.array(front_thw_values) if front_thw_values else np.array([0.0])
    front_thw_ratio = float(np.mean(front_thw_arr > 0.01))
    mean_neighbors = float(np.mean(neighbor_counts)) if neighbor_counts else 0.0

    return {
        "mean_speed": mean_speed,
        "front_thw_ratio": front_thw_ratio,
        "mean_neighbors": mean_neighbors,
        "max_cumulative_yaw_deg": math.degrees(max_cumulative_yaw),
    }


def classify_scene(stats: Dict[str, float]) -> str:
    """Classify a scene into one of the 4 categories based on statistics."""
    if stats["max_cumulative_yaw_deg"] > 30:
        return "intersection"
    if stats["mean_speed"] > 8 and stats["front_thw_ratio"] > 0.5 and stats["mean_neighbors"] < 5:
        return "highway"
    if stats["mean_neighbors"] >= 5 and stats["mean_speed"] < 8:
        return "dense"
    if stats["mean_neighbors"] < 2 and stats["mean_speed"] >= 3:
        return "sparse"
    return "sparse"


def classify_all_scenes(pkl_dir: str) -> Dict[str, List[Tuple[str, Dict[str, float], List[dict]]]]:
    """Load all pkl files and classify each scene.

    Returns: {category: [(scene_name, stats, scene_data), ...]}
    """
    results: Dict[str, list] = {c: [] for c in CATEGORIES}

    if not os.path.isdir(pkl_dir):
        print(f"Feature directory not found: {pkl_dir}")
        return results

    for fname in sorted(os.listdir(pkl_dir)):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(pkl_dir, fname)
        scene_name = fname.replace("_irl_features.pkl", "")
        with open(path, "rb") as f:
            scene_data = pickle.load(f)

        stats = compute_scene_stats(scene_data)
        cat = classify_scene(stats)
        results[cat].append((scene_name, stats, scene_data))

    return results


def select_representatives(
    classified: Dict[str, List[Tuple[str, Dict[str, float], List[dict]]]]
) -> Dict[str, Tuple[str, List[dict]]]:
    """Pick the most typical scene per category."""
    reps = {}
    for cat in CATEGORIES:
        candidates = classified[cat]
        if not candidates:
            continue

        if cat == "intersection":
            best = max(candidates, key=lambda x: x[1]["max_cumulative_yaw_deg"])
        elif cat == "highway":
            best = max(candidates, key=lambda x: x[1]["mean_speed"])
        elif cat == "dense":
            best = max(candidates, key=lambda x: x[1]["mean_neighbors"])
        else:
            best = min(candidates, key=lambda x: x[1]["mean_neighbors"])

        reps[cat] = (best[0], best[2])
    return reps


# ------------------------------------------------------------------ #
# Weight network inference                                           #
# ------------------------------------------------------------------ #

def load_weight_network(pt_path: str, device: torch.device) -> Tuple[WeightNetwork, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a saved .pt weight network checkpoint."""
    ckpt = torch.load(pt_path, map_location=device)
    wn_cfg = ckpt.get("weight_network_config", {})
    history_frames = ckpt.get("history_num_frames", 20)
    num_steps = wn_cfg.get("num_history_steps", history_frames + 1) if wn_cfg else history_frames + 1

    net = WeightNetwork(
        feature_dim=len(ckpt.get("feature_names", FEATURE_NAMES)),
        map_feat_dim=wn_cfg.get("map_feat_dim", 128) if wn_cfg else 128,
        nbr_feat_dim=wn_cfg.get("nbr_feat_dim", 128) if wn_cfg else 128,
        ego_feat_dim=wn_cfg.get("ego_feat_dim", 64) if wn_cfg else 64,
        mlp_hidden=wn_cfg.get("mlp_hidden", []) if wn_cfg else [],
        num_history_steps=num_steps,
        map_channels=wn_cfg.get("map_channels", 3) if wn_cfg else 3,
        map_image_hw=wn_cfg.get("map_image_hw", 224) if wn_cfg else 224,
        map_arch=wn_cfg.get("map_arch", "resnet18") if wn_cfg else "resnet18",
        use_map=wn_cfg.get("use_map", True) if wn_cfg else True,
        use_neighbors=wn_cfg.get("use_neighbors", True) if wn_cfg else True,
        use_ego=wn_cfg.get("use_ego", True) if wn_cfg else True,
        weight_scale=wn_cfg.get("weight_scale", None) if wn_cfg else None,
    )
    net.load_state_dict(ckpt["weight_net_state_dict"])
    net.to(device)
    net.eval()

    norm_mean = ckpt.get("norm_mean")
    norm_std = ckpt.get("norm_std")
    return net, norm_mean, norm_std


def context_to_torch(context: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert numpy context dict to single-row torch tensors."""
    out = {}
    for key, arr in context.items():
        if arr is None:
            continue
        t = torch.as_tensor(np.asarray(arr)).float().to(device)
        if t.dim() > 0 and t.shape[0] > 1:
            t = t[0:1]
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        out[key] = t
    return out


def infer_weights_for_scene(
    net: WeightNetwork, scene_data: List[dict], device: torch.device,
    scene_name: str = "",
) -> np.ndarray:
    """Run weight network on every frame of a scene.

    Returns: (num_frames, 7) array of weight vectors.
    """
    weights = []
    n_ctx = 0
    n_no_ctx = 0
    with torch.no_grad():
        for frame_entry in scene_data:
            ctx = frame_entry.get("context")
            if ctx is None:
                weights.append(np.zeros(len(FEATURE_NAMES)))
                n_no_ctx += 1
                continue
            n_ctx += 1
            ctx_t = context_to_torch(ctx, device)
            w = net(ctx_t).cpu().numpy().squeeze()
            weights.append(w)
    arr = np.array(weights) if weights else np.zeros((0, len(FEATURE_NAMES)))
    print(f"  [{scene_name}] frames={len(scene_data)}, with_context={n_ctx}, "
          f"no_context={n_no_ctx}, weight_shape={arr.shape}")
    if arr.size > 0:
        print(f"    weight range: min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}")
        if n_ctx > 0:
            print(f"    first frame context keys: {list(scene_data[0].get('context', {}).keys())}")
    return arr


# ------------------------------------------------------------------ #
# Visualization                                                      #
# ------------------------------------------------------------------ #

def plot_weight_comparison(
    representatives: Dict[str, Tuple[str, List[dict]]],
    net: WeightNetwork,
    fixed_weights: Optional[np.ndarray],
    device: torch.device,
    save_path: str,
):
    """Generate comparison figure: weight network vs fixed weights."""
    cats_present = [c for c in CATEGORIES if c in representatives]
    n_cats = len(cats_present)
    if n_cats == 0:
        print("No representative scenes found. Nothing to plot.")
        return

    n_rows = 2 if fixed_weights is not None else 1
    fig, axes = plt.subplots(n_rows, n_cats, figsize=(4.5 * n_cats, 3.5 * n_rows),
                             squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(FEATURE_NAMES)))

    for col, cat in enumerate(cats_present):
        scene_name, scene_data = representatives[cat]
        w_arr = infer_weights_for_scene(net, scene_data, device, scene_name)
        if w_arr.shape[0] == 0:
            continue
        frames = np.arange(1, len(w_arr) + 1)

        ax = axes[0, col]
        for fi, fname in enumerate(FEATURE_NAMES):
            ax.plot(frames, w_arr[:, fi], linewidth=1.2, color=colors[fi], label=fname)
        ax.set_title(f"{CATEGORY_LABELS[cat]}\n({scene_name})", fontsize=9)
        ax.set_xlabel("Frame")
        if col == 0:
            ax.set_ylabel("Weight (network)")
        ax.grid(True, alpha=0.3)

        if fixed_weights is not None:
            ax_fix = axes[1, col]
            for fi, fname in enumerate(FEATURE_NAMES):
                ax_fix.axhline(y=fixed_weights[fi], linewidth=1.2, color=colors[fi], label=fname)
            ax_fix.set_xlabel("Frame")
            ax_fix.set_xlim(frames[0], frames[-1])
            if col == 0:
                ax_fix.set_ylabel("Weight (fixed)")
            ax_fix.set_title("Fixed Weights", fontsize=9)
            ax_fix.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(FEATURE_NAMES),
               fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation figure to {save_path}")


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_script_dir, "irl_output")
    feature_dir = os.path.join(out_dir, "features")

    # --- Classify scenes ---
    print("Classifying scenes...")
    classified = classify_all_scenes(feature_dir)
    for cat in CATEGORIES:
        names = [x[0] for x in classified[cat]]
        print(f"  {CATEGORY_LABELS[cat]:25s}: {len(names)} scenes")
        for n, stats, _ in classified[cat]:
            print(f"    {n}: speed={stats['mean_speed']:.1f} m/s, "
                  f"nbr={stats['mean_neighbors']:.1f}, "
                  f"front_thw_ratio={stats['front_thw_ratio']:.2f}, "
                  f"yaw={stats['max_cumulative_yaw_deg']:.1f} deg")

    representatives = select_representatives(classified)
    print(f"\nSelected representatives: {len(representatives)} categories")
    for cat, (name, _) in representatives.items():
        print(f"  {CATEGORY_LABELS[cat]}: {name}")

    # --- Load weight network ---
    pt_files = [f for f in os.listdir(out_dir) if f.endswith(".pt")]
    if not pt_files:
        print(f"No .pt file found in {out_dir}. Run IRL training first.")
        return
    pt_path = os.path.join(out_dir, pt_files[0])
    print(f"\nLoading weight network from {pt_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, norm_mean, norm_std = load_weight_network(pt_path, device)

    # --- Load fixed weights if available ---
    fixed_weights = None
    fixed_pkl = [f for f in os.listdir(out_dir) if f.endswith(".pkl") and "fixed" in f.lower()]
    if fixed_pkl:
        with open(os.path.join(out_dir, fixed_pkl[0]), "rb") as f:
            fixed_data = pickle.load(f)
        if isinstance(fixed_data, dict) and "theta" in fixed_data:
            fixed_weights = np.asarray(fixed_data["theta"])
            print(f"Loaded fixed weights from {fixed_pkl[0]}")

    # --- Visualize ---
    save_path = os.path.join(out_dir, "weight_evaluation.png")
    plot_weight_comparison(representatives, net, fixed_weights, device, save_path)


if __name__ == "__main__":
    main()
