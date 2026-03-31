"""
Adversarial Diffusion Training with Situation-Aware IRL v2.

G step: Use v2 neural reward model to compute context-dependent guidance weights,
        then run guided diffusion rollouts.
D step: Retrain v2 reward model to better distinguish expert vs generated trajectories.

Usage:
    python -m DiffusionGan.adversarialDiffusion_v2
    python -m DiffusionGan.adversarialDiffusion_v2 --model_path ./path/to/v2.pt --num_iterations 50
"""

import numpy as np
import torch
import pickle
import os
import argparse
from typing import Optional, Dict, List, Any

from MaxEntIRL.extract_features_context_v2 import ContextAwareFeatureExtractorV2
from MaxEntIRL.run_irl_context_v2 import SituationAwareIRLV2, ContextIRLDatasetV2
from MaxEntIRL.models_v2 import SituationAwareRewardModelV2, ModelConfig
from MaxEntIRL.irl_config import default_config
from tbsim.configs.scene_edit_config import SceneEditingConfig


class AdversarialIRLDiffusionV2:
    """
    Adversarial training with situation-aware (neural network) reward model.

    Key difference from v1 (linear):
      - v1: fixed θ → reward = θ · φ(s,a)
      - v2: neural w(context) → reward = w(map, neighbors) · φ(s,a)

    For the G step, we pre-compute per-agent weights from the v2 model
    using cached contexts, then average them to create a global guidance
    that works with the existing LearnedRewardGuidance infrastructure.
    """

    def __init__(self, config, model_path: Optional[str] = None):
        self.config = config
        self.extractor: Optional[ContextAwareFeatureExtractorV2] = None
        self.env = None
        self.policy = None
        self.policy_model = None

        # v2 reward model
        self.reward_trainer: Optional[SituationAwareIRLV2] = None
        self.model_path = model_path or os.path.join(
            config.output_dir, "situation_aware_irl_v2.pt"
        )

        # Cached from previous iteration for computing guidance
        self.cached_features: Optional[List[Any]] = None
        self.current_avg_weights: Optional[np.ndarray] = None
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None

        self.training_history = []

        # Wandb
        if self.config.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name or "adversarial-v2",
                tags=self.config.wandb_tags + ["v2", "situation-aware"],
                config={
                    "num_iterations": self.config.num_iterations,
                    "guidance_weight": self.config.guidance_weight,
                    "feature_names": self.config.feature_names,
                    "model_type": "SituationAwareRewardModelV2",
                },
                reinit=True,
            )
            print("Wandb initialized")
        except Exception as e:
            print(f"Wandb init failed: {e}, continuing without wandb")
            self.config.use_wandb = False

    def setup_environment(self):
        """Setup environment using ContextAwareFeatureExtractorV2."""
        cfg = SceneEditingConfig(registered_name="trajdata_nusc_diff")
        if hasattr(self.config, "eval_class"):
            cfg.eval_class = self.config.eval_class
        if hasattr(self.config, "env"):
            cfg.env = self.config.env
        if hasattr(cfg, "edits") and hasattr(cfg.edits, "editing_source"):
            if not isinstance(cfg.edits.editing_source, list):
                cfg.edits.editing_source = [cfg.edits.editing_source]
        for k in cfg[cfg.env]:
            cfg[k] = cfg[cfg.env][k]
        cfg.pop("nusc", None)
        cfg.pop("trajdata", None)
        cfg.ckpt.policy.ckpt_dir = self.config.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = self.config.policy_ckpt_key
        cfg.results_dir = self.config.output_dir

        # Use v2 context-aware extractor
        self.extractor = ContextAwareFeatureExtractorV2(cfg, self.config)
        self.env = self.extractor.env
        self.policy = self.extractor.policy
        self.policy_model = self.extractor.policy_model
        print("Environment setup complete (v2 context-aware extractor)")

    def load_pretrained_reward_model(self):
        """Load the pre-trained situation-aware IRL v2 model."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"V2 model not found: {self.model_path}\n"
                "Train it first: python -m MaxEntIRL.run_irl_context_v2"
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.model_path, map_location=device)

        model_config = checkpoint.get("model_config", ModelConfig())
        self.reward_trainer = SituationAwareIRLV2(
            feature_names=checkpoint.get("feature_names", self.config.feature_names),
            model_config=model_config,
            lr=1e-4,
            weight_decay=1e-4,
            device=device,
        )
        self.reward_trainer.model.load_state_dict(checkpoint["model_state_dict"])
        self.norm_mean = checkpoint.get("norm_mean")
        self.norm_std = checkpoint.get("norm_std")

        print(f"Loaded v2 reward model from {self.model_path}")

    # ------------------------------------------------------------------ #
    #                        G STEP: Generator                            #
    # ------------------------------------------------------------------ #

    def compute_avg_weights_from_contexts(self, features_dict: Dict) -> np.ndarray:
        """
        Run the v2 model on cached contexts to get per-agent weights,
        then average them to produce a global weight vector.
        """
        model = self.reward_trainer.model
        device = self.reward_trainer.device
        model.eval()

        all_weights = []

        for scene_key, scene_entries in features_dict.items():
            for frame_entry in scene_entries:
                ff = frame_entry.get("frame_features", {})
                agent_contexts = ff.get("agent_contexts", {})

                for agent_id, ctx in agent_contexts.items():
                    if not ctx.get("valid", False):
                        continue
                    map_img = ctx.get("map_image")
                    if map_img is None or np.abs(map_img).sum() < 1.0:
                        continue

                    # Prepare inputs
                    map_t = torch.tensor(map_img, dtype=torch.float32).unsqueeze(0).to(device)

                    # Neighbor trajectories
                    neighbor_traj = np.zeros((10, 20, 4), dtype=np.float32)
                    traj_mask = np.zeros(10, dtype=np.float32)
                    if ctx.get("neighbor_trajectories") is not None:
                        for i, nt in enumerate(ctx["neighbor_trajectories"][:10]):
                            nt = np.array(nt, dtype=np.float32)
                            t_len = min(len(nt), 20)
                            f_dim = min(nt.shape[1] if nt.ndim > 1 else 1, 4)
                            if nt.ndim == 1:
                                nt = nt.reshape(-1, 1)
                            neighbor_traj[i, :t_len, :f_dim] = nt[:t_len, :f_dim]
                            traj_mask[i] = 1.0

                    traj_t = torch.tensor(neighbor_traj, dtype=torch.float32).unsqueeze(0).to(device)
                    mask_t = torch.tensor(traj_mask, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        w = model(map_t, traj_t, mask_t)  # (1, num_features)
                    all_weights.append(w.cpu().numpy()[0])

        if not all_weights:
            print("WARNING: no valid contexts found, using zero weights")
            return np.zeros(len(self.config.feature_names), dtype=np.float32)

        avg_w = np.mean(np.stack(all_weights), axis=0)
        print(f"Computed avg weights from {len(all_weights)} agent contexts: {avg_w.round(4)}")
        return avg_w

    def set_guidance_from_weights(self, weights: np.ndarray):
        """Inject averaged weights as learned_reward_guidance into eval_cfg."""
        reward_guidance = {
            "name": "learned_reward_guidance",
            "weight": self.config.guidance_weight,
            "params": {
                "reward_weights": weights.tolist(),
                "feature_names": self.config.feature_names,
                "dt": self.config.step_time,
                "norm_mean": self.norm_mean.tolist() if self.norm_mean is not None else None,
                "norm_std": self.norm_std.tolist() if self.norm_std is not None else None,
            },
            "agents": None,
        }

        eval_cfg = self.env.eval_cfg
        eval_cfg.apply_guidance = True

        if not hasattr(eval_cfg.edits, "guidance_config") or eval_cfg.edits.guidance_config is None:
            eval_cfg.edits.guidance_config = []
        if len(eval_cfg.edits.guidance_config) > 0 and isinstance(eval_cfg.edits.guidance_config[0], list):
            scenes_cfg = eval_cfg.edits.guidance_config
        else:
            scenes_cfg = [eval_cfg.edits.guidance_config]

        new_scenes_cfg = []
        for scene_list in scenes_cfg:
            filtered = [g for g in scene_list if g.get("name") != "learned_reward_guidance"]
            filtered.append(reward_guidance)
            new_scenes_cfg.append(filtered)
        eval_cfg.edits.guidance_config = new_scenes_cfg

        self.current_avg_weights = weights
        print(f"Applied guidance: weights={weights.round(4)}, guidance_weight={self.config.guidance_weight}")

    def generate_trajectories(self) -> Dict:
        """G step: generate trajectories with current guidance, extract features+context."""
        print("G step: generating trajectories with current guidance...")
        features = self.extractor.extract_irl_features_from_all_frames()
        return features

    # ------------------------------------------------------------------ #
    #                        D STEP: Discriminator                        #
    # ------------------------------------------------------------------ #

    def retrain_reward_model(self, features_dict: Dict, n_epochs: int = 30,
                             batch_size: int = 16):
        """D step: retrain v2 reward model on new features+contexts."""
        print("D step: retraining v2 reward model...")

        # Convert features_dict to list format expected by SituationAwareIRLV2
        features_list = list(features_dict.values())

        training_log = self.reward_trainer.fit(
            features_list,
            n_epochs=n_epochs,
            batch_size=batch_size,
            val_split=0.2,
            log_interval=10,
            early_stop_patience=15,
        )

        # Update norm stats
        self.norm_mean = self.reward_trainer.norm_mean
        self.norm_std = self.reward_trainer.norm_std

        return training_log

    # ------------------------------------------------------------------ #
    #                        MAIN TRAINING LOOP                           #
    # ------------------------------------------------------------------ #

    def train_adversarial(self, num_iterations: int = 50, d_epochs: int = 30,
                          d_batch_size: int = 16):
        """
        Main adversarial training loop.

        Args:
            num_iterations: number of G-D alternation rounds
            d_epochs: epochs for D step (reward model retraining) per iteration
            d_batch_size: batch size for D step
        """
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Adversarial Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")

            # --- G step: compute guidance & generate trajectories ---
            if iteration == 0 and self.cached_features is None:
                # First iteration: run without guidance to get initial features
                print("First iteration: running without IRL guidance to get baseline features...")
                features = self.generate_trajectories()
                self.cached_features = features

                # Now compute weights from the pre-trained model
                avg_weights = self.compute_avg_weights_from_contexts(features)
                self.set_guidance_from_weights(avg_weights)

                # Re-generate with guidance
                print("Re-generating with IRL guidance...")
                features = self.generate_trajectories()
            else:
                # Use v2 model weights from previous D step
                avg_weights = self.compute_avg_weights_from_contexts(self.cached_features)
                self.set_guidance_from_weights(avg_weights)
                features = self.generate_trajectories()

            self.cached_features = features

            # --- D step: retrain reward model ---
            d_log = self.retrain_reward_model(features, n_epochs=d_epochs,
                                              batch_size=d_batch_size)

            # --- Evaluate & log ---
            self._evaluate_iteration(iteration, features, d_log)

            # --- Save checkpoint ---
            self._save_checkpoint(iteration)

        print(f"\n{'='*60}")
        print("Adversarial training complete!")
        print(f"{'='*60}")

    def _evaluate_iteration(self, iteration: int, features: Dict,
                            d_log: Optional[Dict]):
        """Log metrics for this iteration."""
        metrics = {
            "iteration": iteration,
            "avg_weights": self.current_avg_weights.tolist() if self.current_avg_weights is not None else None,
            "weight_magnitude": float(np.linalg.norm(self.current_avg_weights)) if self.current_avg_weights is not None else 0,
        }

        # D step metrics
        if d_log and d_log.get("val_loss"):
            metrics["d_val_loss"] = d_log["val_loss"][-1]
            metrics["d_val_expert_prob"] = d_log["val_expert_prob"][-1]
            metrics["d_val_top1_acc"] = d_log["val_top1_acc"][-1]

        # Trajectory quality
        quality = self._compute_quality_metrics(features)
        metrics.update(quality)
        self.training_history.append(metrics)

        # Wandb logging
        if self.config.use_wandb:
            try:
                import wandb
                log_dict = {
                    "iteration": iteration,
                    "reward/weight_magnitude": metrics["weight_magnitude"],
                    "quality/collision_rate": quality.get("collision_rate", 0),
                    "quality/expert_similarity": quality.get("expert_similarity", 0),
                    "quality/diversity": quality.get("diversity", 0),
                }
                if d_log and d_log.get("val_loss"):
                    log_dict["d_step/val_loss"] = d_log["val_loss"][-1]
                    log_dict["d_step/val_expert_prob"] = d_log["val_expert_prob"][-1]
                    log_dict["d_step/val_top1_acc"] = d_log["val_top1_acc"][-1]
                if self.current_avg_weights is not None:
                    for i, (w, name) in enumerate(zip(self.current_avg_weights, self.config.feature_names)):
                        log_dict[f"theta/{name}"] = float(w)
                wandb.log(log_dict)
            except Exception:
                pass

        print(f"Iter {iteration}: weight_mag={metrics['weight_magnitude']:.4f}, "
              f"coll_rate={quality.get('collision_rate', 0):.4f}, "
              f"expert_sim={quality.get('expert_similarity', 0):.4f}")

    def _compute_quality_metrics(self, features_dict: Dict) -> Dict:
        """Compute trajectory quality metrics from features."""
        feat_names = self.config.feature_names
        threshold = getattr(self.config, "collision_threshold", 2.0)

        total_agent_time = 0
        total_collisions = 0
        expert_dists = []
        diversity_dists = []

        for scene_entries in features_dict.values():
            for frame_entry in scene_entries:
                ff = frame_entry.get("frame_features", {})
                agent_rollout = ff.get("agent_rollout_features", {})
                agent_gt = ff.get("agent_ground_truth_features", {})

                for agent_id, rollout_feat_list in agent_rollout.items():
                    if not rollout_feat_list:
                        continue
                    for feat_dict in rollout_feat_list:
                        if "min_dis" in feat_dict:
                            md = np.asarray(feat_dict["min_dis"])
                            if md.size > 0:
                                total_collisions += int(np.sum(md < threshold))
                                total_agent_time += int(md.size)

                    gt_feat = agent_gt.get(agent_id)
                    rollout_vecs = [self._feat_to_vec(fd, feat_names) for fd in rollout_feat_list]
                    if gt_feat is not None and rollout_vecs:
                        gt_vec = self._feat_to_vec(gt_feat, feat_names)
                        dists = [float(np.linalg.norm(rv - gt_vec)) for rv in rollout_vecs]
                        expert_dists.append(np.min(dists))
                    if len(rollout_vecs) >= 2:
                        for i in range(len(rollout_vecs)):
                            for j in range(i + 1, len(rollout_vecs)):
                                diversity_dists.append(float(np.linalg.norm(rollout_vecs[i] - rollout_vecs[j])))

        return {
            "collision_rate": (total_collisions / total_agent_time) if total_agent_time > 0 else 0.0,
            "expert_similarity": float(np.mean(expert_dists)) if expert_dists else 0.0,
            "diversity": float(np.mean(diversity_dists)) if diversity_dists else 0.0,
        }

    @staticmethod
    def _feat_to_vec(feat_dict, feat_names):
        vals = []
        for name in feat_names:
            arr = np.asarray(feat_dict.get(name, []))
            vals.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
        return np.array(vals, dtype=float)

    def _save_checkpoint(self, iteration: int):
        """Save model + training state."""
        weights_dir = os.path.join(self.config.output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Save v2 model
        model_path = os.path.join(
            weights_dir,
            f"situation_aware_v2_{self.config.scene_location}_iter{iteration}.pt",
        )
        self.reward_trainer.save(model_path)

        # Save training history + avg weights (for inference.py compatibility)
        meta_path = os.path.join(
            weights_dir,
            f"{self.config.scene_location}_{iteration}.pkl",
        )
        with open(meta_path, "wb") as f:
            pickle.dump({
                "final_theta": self.current_avg_weights,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
                "iteration": iteration,
                "training_history": self.training_history,
            }, f)

        # Also save as the "latest" for easy loading
        latest_model_path = os.path.join(weights_dir, "situation_aware_v2_latest.pt")
        self.reward_trainer.save(latest_model_path)
        latest_pkl = os.path.join(weights_dir, f"{self.config.scene_location}_99.pkl")
        with open(latest_pkl, "wb") as f:
            pickle.dump({
                "final_theta": self.current_avg_weights,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
                "iteration": iteration,
                "training_history": self.training_history,
            }, f)

        print(f"Checkpoint saved: {model_path}, {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial IRL-Diffusion training with situation-aware v2 reward model"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained situation_aware_irl_v2.pt")
    parser.add_argument("--num_iterations", type=int, default=None,
                        help="Number of adversarial iterations (default: from config)")
    parser.add_argument("--d_epochs", type=int, default=30,
                        help="Epochs per D step (reward model retraining)")
    parser.add_argument("--d_batch_size", type=int, default=16,
                        help="Batch size for D step")
    parser.add_argument("--guidance_weight", type=float, default=None,
                        help="Guidance weight override")
    parser.add_argument("--location", type=str, default=None,
                        choices=["boston", "singapore"],
                        help="Scene location override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override config if specified
    if args.guidance_weight is not None:
        default_config.guidance_weight = args.guidance_weight
    if args.location is not None:
        default_config.scene_location = args.location

    num_iterations = args.num_iterations or default_config.num_iterations

    # Initialize
    trainer = AdversarialIRLDiffusionV2(default_config, model_path=args.model_path)
    trainer.setup_environment()
    trainer.load_pretrained_reward_model()

    # Run adversarial training
    trainer.train_adversarial(
        num_iterations=num_iterations,
        d_epochs=args.d_epochs,
        d_batch_size=args.d_batch_size,
    )

    print("Done!")
