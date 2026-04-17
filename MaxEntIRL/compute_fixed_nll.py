"""Compute per-category NLL for the fixed-weight IRL baseline.

Loads the learned theta from irl_weights.pkl, iterates over all scene
feature pkl files, classifies each scene into one of 4 categories, and
computes mean NLL using the same MaxEnt IRL formula used during training.

Writes: irl_output/fixed_nll_by_category.json

Usage:
    python -m MaxEntIRL.compute_fixed_nll
    python -m MaxEntIRL.compute_fixed_nll --feature-dir /path/to/features
    python -m MaxEntIRL.compute_fixed_nll --weights /path/to/irl_weights.pkl
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_NAMES = [
    "velocity", "a_long", "jerk_long", "a_lateral",
    "front_thw", "left_thw", "right_thw",
]
_NO_NORM = {"front_thw", "left_thw", "right_thw"}
_EPS = 1e-8

CATEGORIES = ["intersection", "highway", "dense", "sparse"]
CATEGORY_LABELS = {
    "intersection": "Intersection",
    "highway":      "Highway Car-Following",
    "dense":        "Dense Traffic",
    "sparse":       "Sparse Road",
}


# ------------------------------------------------------------------ #
# Feature helpers (mirrors run_irl.py:convert_features_to_array)     #
# ------------------------------------------------------------------ #

def _feat_to_vec(feat_dict, norm_mean, norm_std):
    vals = []
    for name in FEATURE_NAMES:
        arr = np.asarray(feat_dict[name])
        vals.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
    vec = np.array(vals, dtype=float)
    if norm_mean is not None and norm_std is not None:
        for i, name in enumerate(FEATURE_NAMES):
            if name not in _NO_NORM:
                vec[i] = (vec[i] - norm_mean[i]) / (norm_std[i] + _EPS)
    return vec


# ------------------------------------------------------------------ #
# NLL computation for one scene                                       #
# ------------------------------------------------------------------ #

def compute_scene_nll(scene_data, theta, norm_mean, norm_std):
    """Return list of per-agent NLL values across all frames of a scene."""
    nlls = []
    for frame_data in scene_data:
        ff = frame_data["frame_features"]
        rollout = ff["agent_rollout_features"]
        gt_feats = ff["agent_ground_truth_features"]

        for agent_id, gt_feat in gt_feats.items():
            if agent_id not in rollout:
                continue

            trajs = []
            for r_feat in rollout[agent_id]:
                r_vec = _feat_to_vec(r_feat, norm_mean, norm_std)
                trajs.append(float(np.dot(r_vec, theta)))

            gt_vec = _feat_to_vec(gt_feat, norm_mean, norm_std)
            trajs.append(float(np.dot(gt_vec, theta)))  # GT always last

            rewards = np.array(trajs, dtype=float)
            max_r = rewards.max()
            exp_r = np.exp(rewards - max_r)
            probs = exp_r / exp_r.sum()
            nlls.append(-np.log(probs[-1] + _EPS))

    return nlls


# ------------------------------------------------------------------ #
# Scene classification (mirrors evaluate_weight_network.py)          #
# ------------------------------------------------------------------ #

def _scene_stats(scene_data):
    all_speeds, front_thw_vals, neighbor_counts = [], [], []
    max_yaw = 0.0

    for frame_entry in scene_data:
        gt = frame_entry["frame_features"]["agent_ground_truth_features"]
        for feat in gt.values():
            v = np.asarray(feat["velocity"])
            if v.size > 0:
                all_speeds.append(float(np.mean(v)))
            f = np.asarray(feat["front_thw"])
            if f.size > 0:
                front_thw_vals.extend(f.tolist())

        ctx = frame_entry.get("context")
        if ctx is not None:
            nbr = ctx.get("all_other_agents_history_positions")
            if nbr is not None:
                arr = np.asarray(nbr)
                if arr.ndim >= 2:
                    neighbor_counts.append(arr.shape[-3] if arr.ndim >= 3 else arr.shape[0])
            yaws = ctx.get("history_yaws")
            if yaws is not None:
                ya = np.asarray(yaws).flatten()
                if ya.size >= 2:
                    max_yaw = max(max_yaw, float(np.abs(ya[-1] - ya[0])))

    mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
    fthw_arr = np.array(front_thw_vals) if front_thw_vals else np.array([0.0])
    return {
        "mean_speed": mean_speed,
        "front_thw_ratio": float(np.mean(fthw_arr > 0.01)),
        "mean_neighbors": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        "max_cumulative_yaw_deg": math.degrees(max_yaw),
    }


def _classify(stats):
    if stats["max_cumulative_yaw_deg"] > 30:
        return "intersection"
    if stats["mean_speed"] > 8 and stats["front_thw_ratio"] > 0.5 and stats["mean_neighbors"] < 5:
        return "highway"
    if stats["mean_neighbors"] >= 5 and stats["mean_speed"] < 8:
        return "dense"
    return "sparse"


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        default=os.path.join(_SCRIPT_DIR, "irl_output", "features"),
        help="Directory containing scene feature pkl files",
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(_SCRIPT_DIR, "irl_output", "irl_weights.pkl"),
        help="Path to irl_weights.pkl",
    )
    args = parser.parse_args()

    # Load theta + normalisation stats
    print(f"Loading weights from {args.weights}")
    with open(args.weights, "rb") as f:
        irl_data = pickle.load(f)
    theta = np.asarray(irl_data["theta"], dtype=float)
    norm_mean = np.asarray(irl_data["norm_mean"]) if irl_data.get("norm_mean") is not None else None
    norm_std = np.asarray(irl_data["norm_std"]) if irl_data.get("norm_std") is not None else None
    print(f"  theta = {np.round(theta, 4)}")

    if not os.path.isdir(args.feature_dir):
        raise FileNotFoundError(
            f"Feature directory not found: {args.feature_dir}\n"
            "Run feature extraction first (extract_features.py)."
        )

    # Load, classify, compute NLL
    nll_by_cat = {c: [] for c in CATEGORIES}
    pkl_files = sorted(f for f in os.listdir(args.feature_dir) if f.endswith(".pkl"))
    print(f"Found {len(pkl_files)} scene pkl files in {args.feature_dir}")

    for fname in pkl_files:
        path = os.path.join(args.feature_dir, fname)
        with open(path, "rb") as f:
            scene_data = pickle.load(f)
        stats = _scene_stats(scene_data)
        cat = _classify(stats)
        scene_nlls = compute_scene_nll(scene_data, theta, norm_mean, norm_std)
        nll_by_cat[cat].extend(scene_nlls)
        print(f"  {fname}: cat={cat}, frames={len(scene_data)}, agents_nll={len(scene_nlls)}")

    # Aggregate and report
    print("\nPer-category NLL (fixed-weight baseline):")
    result = {}
    for cat in CATEGORIES:
        vals = nll_by_cat[cat]
        if vals:
            mean_nll = float(np.mean(vals))
            result[cat] = mean_nll
            print(f"  {CATEGORY_LABELS[cat]:26s}: {mean_nll:.4f}  (n={len(vals)})")
        else:
            print(f"  {CATEGORY_LABELS[cat]:26s}: no data — skipped")

    out_path = os.path.join(_SCRIPT_DIR, "irl_output", "fixed_nll_by_category.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
