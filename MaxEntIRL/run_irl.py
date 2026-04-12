import os
import pickle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

from .irl_config import default_config


class MaxEntIRL:
    """
    MaxEnt IRL with two implementations:

    - Legacy (numpy, fixed theta vector): matches the original behavior.
      Enabled when `use_weight_network=False` or when no per-frame `context`
      is available in the loaded features.
    - Weight network (PyTorch, context-conditional w_theta(c)):
      reward = w_theta(c)^T * phi(s, a).
      Enabled when `use_weight_network=True` and features contain `context`.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_iters: int = 200,
        lr: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lam: float = 0.01,
        seed: int = 42,
        use_weight_network: Optional[bool] = None,
        config=default_config,
        device: Optional[str] = None,
    ):
        self.feature_names = feature_names if feature_names is not None else config.feature_names
        self.feature_num = len(self.feature_names)

        self.n_iters = n_iters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lam = lam
        self.config = config

        # Backwards compatible fixed theta (used when weight network is off
        # or for A/B comparisons).
        self.theta = np.random.RandomState(seed).normal(0, 0.05, size=self.feature_num)

        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.eps = 1e-8

        # Weight network setup
        self.use_weight_network = (
            config.use_weight_network if use_weight_network is None else bool(use_weight_network)
        )
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.weight_net: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def _ensure_weight_net(self) -> None:
        if self.weight_net is not None:
            return
        from .weight_network import build_weight_network_from_config

        self.weight_net = build_weight_network_from_config(self.config).to(self.device)
        wn_cfg = getattr(self.config, "weight_network", None)
        lr = getattr(wn_cfg, "lr", 3e-4) if wn_cfg is not None else 3e-4
        l2 = getattr(wn_cfg, "l2", 1e-4) if wn_cfg is not None else 1e-4
        self.optimizer = torch.optim.Adam(
            self.weight_net.parameters(), lr=lr, weight_decay=l2
        )

    @staticmethod
    def load_features(feature_dir: str) -> List[Any]:
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory {feature_dir} does not exist.")
        features = []
        for filename in os.listdir(feature_dir):
            path = os.path.join(feature_dir, filename)
            if filename.endswith(".pkl") and os.path.isfile(path):
                with open(path, "rb") as f:
                    features.append(pickle.load(f))
        return features

    def convert_features_to_array(self, features: Any, apply_norm: bool = True) -> np.ndarray:
        """
        Convert a feature dict (time-series) to a fixed-length vector in self.feature_names order.
        Aggregation = mean over time. normalized by z-score.
        """
        no_norm_features = {'front_thw', 'left_thw', 'right_thw'}

        if isinstance(features, dict):
            vals = []
            for name in self.feature_names:
                arr = np.asarray(features[name])
                vals.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
            vec = np.array(vals, dtype=float)
        else:
            vec = np.asarray(features, dtype=float)

        if apply_norm and self.norm_mean is not None and self.norm_std is not None:
            normalized_vec = vec.copy()
            for i, feature_name in enumerate(self.feature_names):
                if feature_name not in no_norm_features:
                    normalized_vec[i] = (vec[i] - self.norm_mean[i]) / (self.norm_std[i] + self.eps)
            vec = normalized_vec
        return vec

    def _compute_normalization_stats(self, features: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Collect feature vectors (expert + rollouts) and compute per-feature mean/std."""
        vecs: list[np.ndarray] = []
        for scene_data in features:
            for frame_data in scene_data:
                ff = frame_data["frame_features"]
                roll = ff["agent_rollout_features"]
                gt = ff["agent_ground_truth_features"]
                for _, gt_feat in gt.items():
                    vecs.append(self.convert_features_to_array(gt_feat, apply_norm=False))
                for _, lst in roll.items():
                    for rfeat in lst:
                        vecs.append(self.convert_features_to_array(rfeat, apply_norm=False))

        mat = np.stack(vecs, axis=0)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    def _traj_reward(self, feat_vec: np.ndarray) -> float:
        return float(np.dot(feat_vec, self.theta))

    # -------------------------------------------------------------- #
    # Feature dataset assembly                                       #
    # -------------------------------------------------------------- #

    def _iter_frame_entries(self, features: List[Any]):
        """Flatten nested features list -> iterable of (scene_data, frame_entry)."""
        for scene_data in features:
            for frame_entry in scene_data:
                yield frame_entry

    def _has_context(self, features: List[Any]) -> bool:
        for frame_entry in self._iter_frame_entries(features):
            if "context" in frame_entry and frame_entry["context"]:
                return True
        return False

    def _context_to_torch(self, context: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert a cached context snapshot (numpy) to batch-major torch tensors.
        The extractor saves one snapshot per (scene, frame) capturing the
        trajdata agent batch; here we collapse the agent-batch dim and keep a
        single representative row (the ego of that frame) so the same weight
        vector is shared by all agents in the frame.
        """
        out: Dict[str, torch.Tensor] = {}
        for key, arr in context.items():
            if arr is None:
                continue
            t = torch.as_tensor(np.asarray(arr)).float().to(self.device)
            # The snapshot comes from trajdata AgentBatch where dim 0 is agents
            # in the scene. Take the first row as the scene-level context.
            if t.dim() > 0 and t.shape[0] > 1:
                t = t[0:1]
            elif t.dim() > 0 and t.shape[0] == 1:
                t = t  # already single-row
            else:
                t = t.unsqueeze(0)
            out[key] = t
        return out

    # -------------------------------------------------------------- #
    # Training loop                                                  #
    # -------------------------------------------------------------- #

    def fit(self, features: List[Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Run MaxEnt IRL on the provided features list. Returns either the
        learned theta vector (legacy) or the weight network state dict
        (context-conditional mode).
        """
        # Compute normalization once before iterations
        if self.norm_mean is None or self.norm_std is None:
            self.norm_mean, self.norm_std = self._compute_normalization_stats(features)

        if self.use_weight_network and self._has_context(features):
            return self._fit_weight_network(features)
        if self.use_weight_network:
            print(
                "[MaxEntIRL] use_weight_network=True but no 'context' field "
                "found in features; falling back to legacy theta training."
            )
        return self._fit_legacy(features)

    def _fit_legacy(self, features: List[Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Original numpy + hand-rolled Adam implementation (unchanged math)."""
        theta = self.theta.copy()
        pm = None
        pv = None

        training_log = {
            "iteration": [],
            "average_feature_difference": [],
            "average_log-likelihood": [],
            "average_human_likeness": [],
            "theta": [],
        }

        for it in range(self.n_iters):
            print(f"iteration: {it + 1}/{self.n_iters}")

            feature_exp = np.zeros(self.feature_num, dtype=float)
            human_feature_exp = np.zeros(self.feature_num, dtype=float)
            log_like_list: List[float] = []
            iteration_human_likeness: List[float] = []
            num_demo_agents = 0

            for scene_data in features:
                for frame_data in scene_data:
                    frame_features = frame_data["frame_features"]
                    agent_rollout_features: Dict[Any, List[Dict[str, Any]]] = frame_features["agent_rollout_features"]
                    agent_gt_features: Dict[Any, Dict[str, Any]] = frame_features["agent_ground_truth_features"]

                    for agent_id, gt_features in agent_gt_features.items():
                        if agent_id not in agent_rollout_features:
                            continue

                        agent_trajs: List[Tuple[float, np.ndarray]] = []
                        for rollout_feat_dict in agent_rollout_features[agent_id]:
                            r_vec = self.convert_features_to_array(rollout_feat_dict)
                            r_rew = float(np.dot(r_vec, theta))
                            agent_trajs.append((r_rew, r_vec))

                        gt_vec = self.convert_features_to_array(gt_features)
                        gt_rew = float(np.dot(gt_vec, theta))
                        agent_trajs.append((gt_rew, gt_vec))

                        if not agent_trajs:
                            continue
                        num_demo_agents += 1

                        rewards = np.array([rw for rw, _ in agent_trajs], dtype=float)
                        max_reward = np.max(rewards)
                        exp_rewards = np.exp(rewards - max_reward)
                        probs = exp_rewards / np.sum(exp_rewards)
                        traj_features = np.stack([vec for _, vec in agent_trajs], axis=0)

                        feature_exp += np.dot(probs, traj_features)
                        log_like_list.append(np.log(probs[-1] + self.eps))

                        top_k = min(3, len(probs))
                        top_idx = np.argsort(probs)[-top_k:][::-1]
                        for idx in top_idx:
                            iteration_human_likeness.append(
                                float(np.linalg.norm(traj_features[idx] - gt_vec, ord=2))
                            )

                        human_feature_exp += gt_vec

            if num_demo_agents == 0:
                print("No trajectories found in this iteration")
                continue

            grad = (human_feature_exp - feature_exp) / num_demo_agents - 2.0 * self.lam * theta

            if pm is None:
                pm = np.zeros_like(grad)
                pv = np.zeros_like(grad)
            pm = self.beta1 * pm + (1 - self.beta1) * grad
            pv = self.beta2 * pv + (1 - self.beta2) * (grad * grad)
            mhat = pm / (1 - self.beta1 ** (it + 1))
            vhat = pv / (1 - self.beta2 ** (it + 1))
            theta += self.lr * (mhat / (np.sqrt(vhat) + self.eps))

            training_log["iteration"].append(it + 1)
            training_log["average_feature_difference"].append(
                float(np.linalg.norm((human_feature_exp - feature_exp) / num_demo_agents))
            )
            training_log["average_log-likelihood"].append(
                float(np.mean(log_like_list)) if log_like_list else float("nan")
            )
            training_log["average_human_likeness"].append(
                float(np.mean(iteration_human_likeness)) if iteration_human_likeness else float("nan")
            )
            training_log["theta"].append(theta.copy())

            if (it + 1) % 10 == 0:
                print(f"Iteration {it + 1}: Log-likelihood = {training_log['average_log-likelihood'][-1]:.4f}")

        self.theta = theta
        return theta, training_log

    def _sync_map_shape_from_features(self, features: List[Any]) -> None:
        """
        Peek at the first frame that has a context and copy the actual raster
        shape (C, H, W) and history length T into self.config.weight_network so
        the encoder is built to match the real data instead of the
        (3, 224, 224, T=history_num_frames+1) defaults.
        """
        wn_cfg = getattr(self.config, "weight_network", None)
        if wn_cfg is None:
            return
        for scene_data in features:
            for frame_entry in scene_data:
                ctx = frame_entry.get("context") if isinstance(frame_entry, dict) else None
                if not ctx:
                    continue

                changed = False

                # --- Map raster (C, H, W) ---
                img = ctx.get("image")
                if img is not None:
                    arr = np.asarray(img)
                    # trajdata stores image as (N_agents, C, H, W); drop leading
                    # dims until we reach (C, H, W).
                    while arr.ndim > 3:
                        arr = arr[0]
                    if arr.ndim == 3:
                        c, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
                        old_c = getattr(wn_cfg, "map_channels", None)
                        old_hw = getattr(wn_cfg, "map_image_hw", None)
                        if old_c != c or old_hw != h:
                            print(
                                f"[MaxEntIRL] Inferred map raster shape from features: "
                                f"C={c}, H={h}, W={w} (was C={old_c}, HW={old_hw})."
                            )
                            wn_cfg.map_channels = c
                            wn_cfg.map_image_hw = h
                            changed = True

                # --- History length T ---
                hist_pos = ctx.get("history_positions")
                if hist_pos is not None:
                    arr = np.asarray(hist_pos)
                    # Expected shape (N_agents, T, 2) -> strip leading dims down to (T, 2).
                    while arr.ndim > 2:
                        arr = arr[0]
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        t_steps = int(arr.shape[0])
                        old_t = getattr(wn_cfg, "num_history_steps", None)
                        if old_t != t_steps:
                            print(
                                f"[MaxEntIRL] Inferred history length from features: "
                                f"T={t_steps} (was num_history_steps={old_t})."
                            )
                            wn_cfg.num_history_steps = t_steps
                            changed = True

                if changed:
                    print("[MaxEntIRL] Rebuilding WeightNetwork with the inferred shapes.")
                    # Invalidate any previously-built encoder so it gets rebuilt
                    # with the new shape.
                    self.weight_net = None
                    self.optimizer = None
                return

    def _fit_weight_network(self, features: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        PyTorch training loop for context-conditional w_theta(c).

        For each (frame, agent) triple with rollouts + expert:
            w = weight_net(context_frame)                      # (1, F)
            rewards_rollouts = sum(w * phi_rollouts, dim=-1)   # (R,)
            reward_expert    = sum(w * phi_expert)             # scalar
            log_Z = logsumexp([rewards_rollouts, reward_expert])
            loss_i = -(reward_expert - log_Z)
        Total loss = mean over (frame, agent) pairs + L2 regularization.
        """
        # Infer the actual BEV raster shape from the first available context so
        # the map encoder is built with the right in_channels (trajdata emits
        # multi-layer rasters, not 3-channel RGB).
        self._sync_map_shape_from_features(features)
        self._ensure_weight_net()

        # Pre-pack all (frame -> torch context, list of (feat_rollouts, feat_expert))
        packed_frames = []
        for scene_data in features:
            for frame_entry in scene_data:
                context = frame_entry.get("context")
                if context is None:
                    continue
                frame_features = frame_entry["frame_features"]
                agent_rollout_features = frame_features["agent_rollout_features"]
                agent_gt_features = frame_features["agent_ground_truth_features"]

                agent_examples = []
                for agent_id, gt_feat in agent_gt_features.items():
                    if agent_id not in agent_rollout_features:
                        continue
                    rollout_vecs = np.stack(
                        [
                            self.convert_features_to_array(rfd)
                            for rfd in agent_rollout_features[agent_id]
                        ],
                        axis=0,
                    ) if agent_rollout_features[agent_id] else np.zeros((0, self.feature_num))
                    gt_vec = self.convert_features_to_array(gt_feat)
                    agent_examples.append((rollout_vecs.astype(np.float32), gt_vec.astype(np.float32)))

                if not agent_examples:
                    continue
                packed_frames.append((context, agent_examples))

        if not packed_frames:
            print("[MaxEntIRL] No frames with context available; cannot train weight network.")
            return {}, {}

        training_log = {
            "iteration": [],
            "loss": [],
            "average_log-likelihood": [],
            "average_human_likeness": [],
            "average_weight_norm": [],
        }

        # Count total agent examples for gradient averaging
        total_examples = sum(len(ae) for _, ae in packed_frames)

        for it in range(self.n_iters):
            self.optimizer.zero_grad()
            running_loss = 0.0
            n_examples = 0
            log_likes = []
            human_likeness = []
            w_norms = []

            for context_np, agent_examples in packed_frames:
                ctx_torch = self._context_to_torch(context_np)
                w = self.weight_net(ctx_torch)  # (1, F)
                w_norms.append(float(w.detach().norm().item()))

                for rollout_vecs_np, gt_vec_np in agent_examples:
                    rollout_vecs = torch.from_numpy(rollout_vecs_np).to(self.device)  # (R, F)
                    gt_vec = torch.from_numpy(gt_vec_np).to(self.device)              # (F,)

                    if rollout_vecs.numel() == 0:
                        stacked = gt_vec.unsqueeze(0)
                    else:
                        stacked = torch.cat([rollout_vecs, gt_vec.unsqueeze(0)], dim=0)  # (R+1, F)

                    rewards = (stacked * w).sum(dim=-1)  # (R+1,)
                    log_z = torch.logsumexp(rewards, dim=0)
                    expert_reward = rewards[-1]
                    nll = -(expert_reward - log_z)

                    # Gradient accumulation: backward each example immediately
                    # to free the computation graph and avoid OOM.
                    (nll / total_examples).backward()
                    running_loss += float(nll.item())
                    n_examples += 1

                    with torch.no_grad():
                        probs = torch.softmax(rewards, dim=0)
                        log_likes.append(float(torch.log(probs[-1] + self.eps).item()))
                        top_k = min(3, probs.numel())
                        _, top_idx = torch.topk(probs, top_k)
                        for idx in top_idx.tolist():
                            human_likeness.append(
                                float(torch.norm(stacked[idx] - gt_vec).item())
                            )

            if n_examples == 0:
                print("No trajectories found in this iteration")
                continue

            torch.nn.utils.clip_grad_norm_(self.weight_net.parameters(), max_norm=5.0)
            self.optimizer.step()

            training_log["iteration"].append(it + 1)
            training_log["loss"].append(running_loss / n_examples)
            training_log["average_log-likelihood"].append(
                float(np.mean(log_likes)) if log_likes else float("nan")
            )
            training_log["average_human_likeness"].append(
                float(np.mean(human_likeness)) if human_likeness else float("nan")
            )
            training_log["average_weight_norm"].append(
                float(np.mean(w_norms)) if w_norms else 0.0
            )

            if (it + 1) % 10 == 0:
                print(
                    f"Iteration {it + 1}: loss={training_log['loss'][-1]:.4f} "
                    f"LL={training_log['average_log-likelihood'][-1]:.4f} "
                    f"|w|={training_log['average_weight_norm'][-1]:.3f}"
                )

        # Return the weight network state dict as the "learned reward" artifact.
        state = {
            "weight_net_state_dict": {
                k: v.detach().cpu() for k, v in self.weight_net.state_dict().items()
            },
        }
        return state, training_log

    # -------------------------------------------------------------- #
    # Persistence                                                    #
    # -------------------------------------------------------------- #

    def save_results(
        self,
        artifact: Any,
        training_log: Dict[str, Any],
        path: str = "irl_results.pkl",
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ) -> None:
        """
        Save IRL results. When the weight network is used, `artifact` is a
        dict containing `weight_net_state_dict`, and the output is a torch
        checkpoint with the weight network config plus normalization stats.
        Otherwise `artifact` is the legacy theta vector and we pickle it.
        """
        if isinstance(artifact, dict) and "weight_net_state_dict" in artifact:
            out_path = path
            if out_path.endswith(".pkl"):
                out_path = out_path[:-4] + ".pt"
            torch.save(
                {
                    "weight_net_state_dict": artifact["weight_net_state_dict"],
                    "training_log": training_log,
                    "norm_mean": np.asarray(norm_mean) if norm_mean is not None else None,
                    "norm_std": np.asarray(norm_std) if norm_std is not None else None,
                    "feature_names": list(self.feature_names),
                    "weight_network_config": self.config.weight_network.__dict__
                    if hasattr(self.config, "weight_network") and self.config.weight_network is not None
                    else None,
                    "history_num_frames": getattr(self.config, "history_num_frames", None),
                },
                out_path,
            )
            print(f"Saved IRL weight network to {out_path}")
            return

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "theta": artifact,
                    "training_log": training_log,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                },
                f,
            )
        print(f"Saved IRL results to {path}")


def plot_training_curves(training_log: Dict[str, Any], save_path: str) -> None:
    """Save a training curve figure from training_log to *save_path*."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    iters = training_log.get("iteration", [])
    if not iters:
        print("[plot] No iterations recorded — skipping plot.")
        return

    # Collect which metrics exist and have plottable data
    candidates = [
        ("loss", "Loss"),
        ("average_log-likelihood", "Avg Log-Likelihood"),
        ("average_human_likeness", "Avg Human Likeness"),
        ("average_weight_norm", "Avg Weight Norm"),
        ("average_feature_difference", "Avg Feature Diff"),
    ]
    panels = [(key, label) for key, label in candidates if key in training_log and training_log[key]]

    if not panels:
        print("[plot] No plottable metrics — skipping plot.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.5 * len(panels)), squeeze=False)
    for ax_row, (key, label) in zip(axes, panels):
        ax = ax_row[0]
        vals = training_log[key]
        ax.plot(iters[: len(vals)], vals, linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curve plot to {save_path}")


if __name__ == "__main__":
    feature_dir = os.path.join(default_config.output_dir, "features")
    features = MaxEntIRL.load_features(feature_dir)

    irl = MaxEntIRL(feature_names=default_config.feature_names, config=default_config)
    artifact, log = irl.fit(features)

    if isinstance(artifact, dict) and "weight_net_state_dict" in artifact:
        print("Final: weight network trained.")
        out_path = os.path.join(default_config.output_dir, "irl_weights.pt")
    else:
        print(f"Final learned weights: {artifact}")
        out_path = os.path.join(default_config.output_dir, "irl_weights.pkl")

    irl.save_results(artifact, log, path=out_path, norm_mean=irl.norm_mean, norm_std=irl.norm_std)

    # Save training curve plot
    plot_path = os.path.join(default_config.output_dir, "training_curves.png")
    plot_training_curves(log, plot_path)
