"""
Evaluation & Visualization: Dynamic (v2) vs Fixed (Linear) IRL Weights

Generates poster-ready figures comparing the situation-aware weight network
against a linear IRL baseline on the same dataset.

Usage:
    cd ~/CTG-Test
    PYTHONPATH=./CTG:./Pplan:$PYTHONPATH python -m MaxEntIRL.evaluate_v2 \
        --model_path ./MaxEntIRL/outputs_boston_50_context/situation_aware_irl_v2.pt \
        --feature_dir ./MaxEntIRL/outputs_boston_50_context/features \
        --output_dir ./MaxEntIRL/evaluation_results
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple

from .models_v2 import SituationAwareRewardModelV2, ModelConfig
from .run_irl import MaxEntIRL
from .run_irl_context_v2 import ContextIRLDatasetV2, load_features
from .irl_config import default_config

# Display labels for the 7 features
FEATURE_LABELS = ['Velocity', 'Accel (Long)', 'Jerk (Long)', 'Accel (Lat)',
                  'Front THW', 'Left THW', 'Right THW']


# ──────────────────────────────────────────────
# 1. Model Loading
# ──────────────────────────────────────────────
def load_v2_model(model_path: str, device: torch.device):
    """Load the trained v2 situation-aware reward model."""
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint.get("model_config", ModelConfig())
    model = SituationAwareRewardModelV2(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded v2 model from {model_path}")
    print(f"  Features: {checkpoint.get('feature_names', default_config.feature_names)}")
    return model, checkpoint


# ──────────────────────────────────────────────
# 2. Linear IRL Baseline
# ──────────────────────────────────────────────
def train_linear_baseline(features: List[Any], feature_names: List[str],
                          n_iters: int = 200) -> Tuple[np.ndarray, Dict]:
    """Train a linear MaxEnt IRL model on the same feature data."""
    print("\n" + "=" * 50)
    print("Training linear IRL baseline ...")
    print("=" * 50)

    irl = MaxEntIRL(feature_names=feature_names, n_iters=n_iters)
    theta, training_log = irl.fit(features)

    print(f"\nLinear theta: {np.round(theta, 4)}")
    return theta, training_log, irl.norm_mean, irl.norm_std


# ──────────────────────────────────────────────
# 3. Evaluate Both Models
# ──────────────────────────────────────────────
def evaluate_both_models(
    model: SituationAwareRewardModelV2,
    linear_theta: np.ndarray,
    dataset: ContextIRLDatasetV2,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate both the dynamic v2 model and the linear baseline on every
    sample in the dataset.  Returns aggregated metrics plus per-sample data
    needed for the visualisation functions.
    """
    model.eval()
    theta_tensor = torch.tensor(linear_theta, dtype=torch.float32, device=device)

    # Accumulators
    v2_nlls, lin_nlls = [], []
    v2_expert_probs, lin_expert_probs = [], []
    v2_top1, lin_top1 = [], []
    v2_top3, lin_top3 = [], []
    all_dynamic_weights = []
    context_metadata = []  # (num_neighbors, ego_speed, scene_idx)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        raw = dataset.samples[idx]

        gt_feat = sample["gt_features"].unsqueeze(0).to(device)        # [1,7]
        roll_feat = sample["rollout_features"].unsqueeze(0).to(device)  # [1,R,7]
        map_img = sample["map_image"].unsqueeze(0).to(device)           # [1,5,224,224]
        neigh_traj = sample["neighbor_trajectories"].unsqueeze(0).to(device)
        traj_mask = sample["traj_mask"].unsqueeze(0).to(device)

        num_rollouts = roll_feat.shape[1]
        all_feat = torch.cat([roll_feat, gt_feat.unsqueeze(1)], dim=1)  # [1,R+1,7]

        with torch.no_grad():
            # ---- V2 (dynamic) ----
            weights = model(map_img, neigh_traj, traj_mask)  # [1,7]
            v2_rewards = (weights.unsqueeze(1) * all_feat).sum(dim=-1)  # [1,R+1]
            v2_log_z = torch.logsumexp(v2_rewards, dim=1)
            v2_gt_reward = v2_rewards[:, -1]
            v2_nll = -(v2_gt_reward - v2_log_z).item()
            v2_probs = torch.softmax(v2_rewards, dim=1)

            # ---- Linear ----
            lin_rewards = (theta_tensor.unsqueeze(0).unsqueeze(0) * all_feat).sum(dim=-1)
            lin_log_z = torch.logsumexp(lin_rewards, dim=1)
            lin_gt_reward = lin_rewards[:, -1]
            lin_nll = -(lin_gt_reward - lin_log_z).item()
            lin_probs = torch.softmax(lin_rewards, dim=1)

        # Metrics
        v2_nlls.append(v2_nll)
        lin_nlls.append(lin_nll)
        v2_expert_probs.append(v2_probs[0, -1].item())
        lin_expert_probs.append(lin_probs[0, -1].item())

        # Top-1
        v2_top1.append(int(v2_probs.argmax(dim=1).item() == num_rollouts))
        lin_top1.append(int(lin_probs.argmax(dim=1).item() == num_rollouts))

        # Top-3
        _, v2_top3_idx = v2_probs.topk(min(3, v2_probs.shape[1]), dim=1)
        _, lin_top3_idx = lin_probs.topk(min(3, lin_probs.shape[1]), dim=1)
        v2_top3.append(int((v2_top3_idx == num_rollouts).any().item()))
        lin_top3.append(int((lin_top3_idx == num_rollouts).any().item()))

        all_dynamic_weights.append(weights[0].cpu().numpy())

        # Context metadata
        ctx = raw.get("context", {})
        num_neighbors = ctx.get("num_neighbors", 0)
        ego_state = ctx.get("ego_state", None)
        ego_speed = float(ego_state[2]) if ego_state is not None and len(ego_state) > 2 else 0.0
        context_metadata.append({
            "num_neighbors": num_neighbors,
            "ego_speed": ego_speed,
            "agent_id": raw.get("agent_id", idx),
        })

    dynamic_weights = np.stack(all_dynamic_weights)  # [N, 7]

    metrics = {
        "v2_nll": float(np.mean(v2_nlls)),
        "lin_nll": float(np.mean(lin_nlls)),
        "v2_expert_prob": float(np.mean(v2_expert_probs)),
        "lin_expert_prob": float(np.mean(lin_expert_probs)),
        "v2_top1": float(np.mean(v2_top1)),
        "lin_top1": float(np.mean(lin_top1)),
        "v2_top3": float(np.mean(v2_top3)),
        "lin_top3": float(np.mean(lin_top3)),
        "dynamic_weights": dynamic_weights,
        "linear_theta": linear_theta,
        "context_metadata": context_metadata,
        "num_samples": len(dataset),
    }
    return metrics


# ──────────────────────────────────────────────
# 4. Visualisation Functions
# ──────────────────────────────────────────────

def plot_metric_comparison(metrics: Dict, save_path: str):
    """Figure 1: Grouped bar chart comparing 4 key metrics."""
    labels = ['NLL Loss', 'Expert Prob', 'Top-1 Acc', 'Top-3 Acc']
    linear_vals = [metrics["lin_nll"], metrics["lin_expert_prob"],
                   metrics["lin_top1"], metrics["lin_top3"]]
    dynamic_vals = [metrics["v2_nll"], metrics["v2_expert_prob"],
                    metrics["v2_top1"], metrics["v2_top3"]]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, linear_vals, width, label='Linear (Fixed)',
                   color='#5B9BD5', edgecolor='white')
    bars2 = ax.bar(x + width / 2, dynamic_vals, width, label='Dynamic (Ours)',
                   color='#ED7D31', edgecolor='white')

    ax.set_ylabel('Value', fontsize=13)
    ax.set_title('Quantitative Comparison: Linear vs Dynamic Weights', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(training_log: Dict, save_path: str):
    """Figure 2: 2x2 training curves from the saved training log."""
    if not training_log or "epoch" not in training_log:
        print("  Skipped training curves (no training_log in checkpoint)")
        return

    epochs = training_log["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Loss
    ax = axes[0, 0]
    ax.plot(epochs, training_log["train_loss"], 'b-', label='Train', lw=2)
    ax.plot(epochs, training_log["val_loss"], 'r--', label='Val', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL Loss')
    ax.set_title('(a) Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Expert Probability
    ax = axes[0, 1]
    ax.plot(epochs, training_log["train_expert_prob"], 'b-', label='Train', lw=2)
    ax.plot(epochs, training_log["val_expert_prob"], 'r--', label='Val', lw=2)
    random_prob = 1.0 / 5  # rough: 4 rollouts + 1 expert
    ax.axhline(y=random_prob, color='gray', ls=':', label=f'Random ({random_prob:.2f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Expert Probability')
    ax.set_title('(b) Expert Selection Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Top-K Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, training_log["train_top1_acc"], 'b-', label='Train Top-1', lw=2)
    ax.plot(epochs, training_log["val_top1_acc"], 'r--', label='Val Top-1', lw=2)
    ax.plot(epochs, training_log["train_top3_acc"], 'g-', label='Train Top-3', lw=1.5)
    ax.plot(epochs, training_log["val_top3_acc"], 'm--', label='Val Top-3', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Top-K Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Weight Variance
    ax = axes[1, 1]
    ax.plot(epochs, training_log["weight_variance"], color='purple', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Variance')
    ax.set_title('(d) Cross-Sample Weight Variance')
    ax.grid(True, alpha=0.3)

    plt.suptitle('V2 Model Training Curves', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_weight_distribution(dynamic_weights: np.ndarray, linear_theta: np.ndarray,
                             feature_names: List[str], save_path: str):
    """
    Figure 3: Violin plots of dynamic weights with linear baseline overlaid.
    One subplot per feature. Shows that dynamic weights VARY while linear is fixed.
    """
    n_feat = len(feature_names)
    fig, axes = plt.subplots(1, n_feat, figsize=(3 * n_feat, 5), sharey=False)

    for i, (ax, label) in enumerate(zip(axes, feature_names)):
        data = dynamic_weights[:, i]
        parts = ax.violinplot(data, positions=[0], showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#ED7D31')
            pc.set_alpha(0.6)
        for key in ('cmins', 'cmaxes', 'cmedians', 'cbars'):
            if key in parts:
                parts[key].set_color('#C55A11')

        # Linear baseline
        ax.axhline(y=linear_theta[i], color='#5B9BD5', ls='--', lw=2,
                    label=f'Linear: {linear_theta[i]:.3f}')
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Weight Distribution: Dynamic (violin) vs Linear (dashed)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_context_sensitivity(dynamic_weights: np.ndarray, context_metadata: List[Dict],
                             feature_names: List[str], save_path: str):
    """
    Figure 4: Show that weights change with traffic context.
    (a) Group by neighbor count  (b) Group by ego speed
    """
    n_feat = len(feature_names)
    num_neighbors = np.array([m["num_neighbors"] for m in context_metadata])
    ego_speeds = np.array([m["ego_speed"] for m in context_metadata])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── (a) By neighbor count ──
    ax = axes[0]
    bins_n = [("Sparse (0-2)", num_neighbors <= 2),
              ("Medium (3-5)", (num_neighbors >= 3) & (num_neighbors <= 5)),
              ("Dense (6+)", num_neighbors >= 6)]

    x = np.arange(n_feat)
    width = 0.25
    offsets = np.linspace(-width, width, len(bins_n))
    colors = ['#70AD47', '#5B9BD5', '#ED7D31']

    for j, (label, mask, offset, color) in enumerate(
            zip([b[0] for b in bins_n], [b[1] for b in bins_n], offsets, colors)):
        if mask.sum() == 0:
            continue
        means = dynamic_weights[mask].mean(axis=0)
        stds = dynamic_weights[mask].std(axis=0)
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=f'{label} (n={mask.sum()})', color=color, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Dynamic Weight', fontsize=12)
    ax.set_title('(a) Weights by Traffic Density (Neighbor Count)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # ── (b) By ego speed ──
    ax = axes[1]
    bins_s = [("Slow (<3 m/s)", ego_speeds < 3),
              ("Medium (3-8 m/s)", (ego_speeds >= 3) & (ego_speeds < 8)),
              ("Fast (>8 m/s)", ego_speeds >= 8)]

    for j, (label, mask, offset, color) in enumerate(
            zip([b[0] for b in bins_s], [b[1] for b in bins_s], offsets, colors)):
        if mask.sum() == 0:
            continue
        means = dynamic_weights[mask].mean(axis=0)
        stds = dynamic_weights[mask].std(axis=0)
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=f'{label} (n={mask.sum()})', color=color, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Dynamic Weight', fontsize=12)
    ax.set_title('(b) Weights by Ego Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Context-Sensitivity Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scene_weight_heatmap(dynamic_weights: np.ndarray, context_metadata: List[Dict],
                              feature_names: List[str], save_path: str,
                              max_scenes: int = 30):
    """
    Figure 5: Heatmap of average dynamic weights per scene.
    Rows = scenes, Columns = 7 features.
    """
    # Group weights by scene (using agent_id prefix or index chunks)
    # Since samples come from sequential scene pkl files, group by fixed-size chunks
    n = len(dynamic_weights)
    if n == 0:
        print("  Skipped heatmap (no samples)")
        return

    # Try to group by num_neighbors pattern or just evenly split into ~scene-sized groups
    # Best approach: create scene groupings from the order of the dataset
    scene_size = max(1, n // max_scenes)
    n_scenes = min(max_scenes, n)

    scene_means = []
    scene_labels = []
    for i in range(0, n, scene_size):
        chunk = dynamic_weights[i:i + scene_size]
        if len(chunk) == 0:
            break
        scene_means.append(chunk.mean(axis=0))
        scene_labels.append(f"Group {len(scene_labels) + 1}")
        if len(scene_labels) >= max_scenes:
            break

    if len(scene_means) < 2:
        print("  Skipped heatmap (not enough scene groups)")
        return

    mat = np.stack(scene_means)  # [S, 7]

    fig, ax = plt.subplots(figsize=(10, max(6, len(mat) * 0.35)))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlBu_r",
                xticklabels=feature_names, yticklabels=scene_labels,
                ax=ax, linewidths=0.5, cbar_kws={"label": "Weight Value"})
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Agent Groups', fontsize=12)
    ax.set_title('Dynamic Weight Patterns Across Agent Groups', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 5. Summary Table
# ──────────────────────────────────────────────
def print_summary_table(metrics: Dict):
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY: Linear (Fixed) vs Dynamic (Ours)")
    print("=" * 65)
    fmt = "  {:<22} {:>12} {:>12} {:>12}"
    print(fmt.format("Metric", "Linear", "Dynamic", "Improvement"))
    print("  " + "-" * 60)

    rows = [
        ("NLL Loss",       metrics["lin_nll"],          metrics["v2_nll"],          True),
        ("Expert Prob",    metrics["lin_expert_prob"],   metrics["v2_expert_prob"],  False),
        ("Top-1 Accuracy", metrics["lin_top1"],          metrics["v2_top1"],         False),
        ("Top-3 Accuracy", metrics["lin_top3"],          metrics["v2_top3"],         False),
    ]

    for name, lin_val, v2_val, lower_is_better in rows:
        if lower_is_better:
            diff_pct = (lin_val - v2_val) / (abs(lin_val) + 1e-8) * 100
            arrow = "+" if diff_pct > 0 else ""
        else:
            diff_pct = (v2_val - lin_val) / (abs(lin_val) + 1e-8) * 100
            arrow = "+" if diff_pct > 0 else ""
        print(fmt.format(name, f"{lin_val:.4f}", f"{v2_val:.4f}",
                         f"{arrow}{diff_pct:.1f}%"))

    print("  " + "-" * 60)
    print(f"  Total samples evaluated: {metrics['num_samples']}")

    # Weight variance (unique to dynamic model)
    w = metrics["dynamic_weights"]
    per_feat_var = w.var(axis=0)
    print(f"\n  Dynamic weight variance per feature:")
    for name, v in zip(FEATURE_LABELS, per_feat_var):
        print(f"    {name:<18} {v:.6f}")
    print(f"    {'Mean variance':<18} {per_feat_var.mean():.6f}")
    print("=" * 65)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic v2 vs linear IRL weights")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to situation_aware_irl_v2.pt")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with feature pkl files")
    parser.add_argument("--output_dir", type=str, default="./MaxEntIRL/evaluation_results",
                        help="Where to save figures")
    parser.add_argument("--linear_iters", type=int, default=200,
                        help="Training iterations for linear baseline")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Load features ──
    features = load_features(args.feature_dir)
    if not features:
        print(f"ERROR: No feature files found in {args.feature_dir}")
        return
    feature_names = default_config.feature_names

    # ── Load v2 model ──
    model, checkpoint = load_v2_model(args.model_path, device)
    training_log = checkpoint.get("training_log", {})

    # ── Train linear baseline ──
    linear_theta, linear_log, lin_norm_mean, lin_norm_std = \
        train_linear_baseline(features, feature_names, n_iters=args.linear_iters)

    # ── Build dataset (reuse v2 data pipeline) ──
    print("\nBuilding evaluation dataset ...")
    dataset = ContextIRLDatasetV2(
        features, feature_names,
        map_channels=model.config.map_channels,
        verbose=True
    )

    # Apply the same normalization that was used during v2 training
    saved_norm_mean = checkpoint.get("norm_mean")
    saved_norm_std = checkpoint.get("norm_std")
    if saved_norm_mean is not None:
        dataset.norm_mean = np.array(saved_norm_mean)
        dataset.norm_std = np.array(saved_norm_std)
        # Re-normalize linear theta with the v2 dataset normalization
        # (linear IRL learns theta on its own normalization, but for fair
        # comparison we must evaluate linear theta on the same normalized features)

    # ── Evaluate both models ──
    print("\nEvaluating both models on full dataset ...")
    metrics = evaluate_both_models(model, linear_theta, dataset, device)

    # ── Print summary ──
    print_summary_table(metrics)

    # ── Generate figures ──
    print("\nGenerating poster figures ...")

    plot_metric_comparison(
        metrics,
        os.path.join(args.output_dir, "fig1_metric_comparison.png"))

    plot_training_curves(
        training_log,
        os.path.join(args.output_dir, "fig2_training_curves.png"))

    plot_weight_distribution(
        metrics["dynamic_weights"], metrics["linear_theta"],
        FEATURE_LABELS,
        os.path.join(args.output_dir, "fig3_weight_distribution.png"))

    plot_context_sensitivity(
        metrics["dynamic_weights"], metrics["context_metadata"],
        FEATURE_LABELS,
        os.path.join(args.output_dir, "fig4_context_sensitivity.png"))

    plot_scene_weight_heatmap(
        metrics["dynamic_weights"], metrics["context_metadata"],
        FEATURE_LABELS,
        os.path.join(args.output_dir, "fig5_scene_heatmap.png"))

    print(f"\nAll figures saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
