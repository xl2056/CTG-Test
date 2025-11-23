import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .irl_config import default_config


class MaxEntIRL:
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
    ):
        # Use feature names from config when not provided
        self.feature_names = feature_names if feature_names is not None else default_config.feature_names
        self.feature_num = len(self.feature_names)

        self.n_iters = n_iters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lam = lam

        self.theta = np.random.RandomState(seed).normal(0, 0.05, size=self.feature_num)

        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.eps = 1e-8
        
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
        # Define which features should NOT be normalized
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
            # Apply normalization selectively
            normalized_vec = vec.copy()
            for i, feature_name in enumerate(self.feature_names):
                if feature_name not in no_norm_features:
                    normalized_vec[i] = (vec[i] - self.norm_mean[i]) / (self.norm_std[i] + self.eps)
                # Proximity features remain unchanged (no normalization)
            vec = normalized_vec
        return vec

    def _compute_normalization_stats(self, features: List[Any]) -> tuple[np.ndarray, np.ndarray]:
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
        std = np.where(std < 1e-6, 1.0, std)  # avoid tiny std
        return mean, std

    def _traj_reward(self, feat_vec: np.ndarray) -> float:
        return float(np.dot(feat_vec, self.theta))

    def fit(self, features: List[Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        features is the list loaded from pkl files:
        [
          [  # scene
            {
              "start_frame": int,
              "frame_features": {
                "agent_rollout_features": {agent_id: [feature_dict, ...], ...},
                "agent_ground_truth_features": {agent_id: feature_dict, ...}
              }
            },
            ...
          ],
          ...
        ]
        """
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

        # Compute normalization once before iterations
        if self.norm_mean is None or self.norm_std is None:
            self.norm_mean, self.norm_std = self._compute_normalization_stats(features)
            
        for it in range(self.n_iters):
            print(f"iteration: {it + 1}/{self.n_iters}")

            # Accumulators
            feature_exp = np.zeros(self.feature_num, dtype=float)
            human_feature_exp = np.zeros(self.feature_num, dtype=float)
            log_like_list: List[float] = []
            iteration_human_likeness: List[float] = []
            num_demo_agents = 0

            # Iterate scenes -> frames -> agents
            for scene_data in features:
                for frame_data in scene_data:
                    frame_features = frame_data["frame_features"]
                    agent_rollout_features: Dict[Any, List[Dict[str, Any]]] = frame_features["agent_rollout_features"]
                    agent_gt_features: Dict[Any, Dict[str, Any]] = frame_features["agent_ground_truth_features"]

                    for agent_id, gt_features in agent_gt_features.items():
                        if agent_id not in agent_rollout_features:
                            continue

                        # Prepare candidate trajectories for this agent: rollouts + expert
                        agent_trajs: List[Tuple[float, np.ndarray]] = []

                        # Rollouts
                        for rollout_feat_dict in agent_rollout_features[agent_id]:
                            r_vec = self.convert_features_to_array(rollout_feat_dict)
                            r_rew = float(np.dot(r_vec, theta))
                            agent_trajs.append((r_rew, r_vec))

                        # Expert (ground truth)
                        gt_vec = self.convert_features_to_array(gt_features)
                        gt_rew = float(np.dot(gt_vec, theta))
                        agent_trajs.append((gt_rew, gt_vec))

                        if not agent_trajs:
                            continue

                        num_demo_agents += 1

                        # Normalize rewards for numerical stability
                        rewards = np.array([rw for rw, _ in agent_trajs], dtype=float)
                        max_reward = np.max(rewards)
                        exp_rewards = np.exp(rewards - max_reward)
                        probs = exp_rewards / np.sum(exp_rewards)

                        traj_features = np.stack([vec for _, vec in agent_trajs], axis=0)

                        # Policy expectation and logs for this agent
                        feature_exp += np.dot(probs, traj_features)
                        log_like_list.append(np.log(probs[-1] + self.eps))

                        # Human-likeness of top-k
                        top_k = min(3, len(probs))
                        top_idx = np.argsort(probs)[-top_k:][::-1]
                        for idx in top_idx:
                            iteration_human_likeness.append(float(np.linalg.norm(traj_features[idx] - gt_vec, ord=2)))

                        # Expert feature expectation
                        human_feature_exp += gt_vec

            if num_demo_agents == 0:
                print("No trajectories found in this iteration")
                continue

            # Gradient of (E_data - E_model) - 2*lam*theta
            grad = (human_feature_exp - feature_exp) / num_demo_agents - 2.0 * self.lam * theta

            # Adam update
            if pm is None:
                pm = np.zeros_like(grad)
                pv = np.zeros_like(grad)
            pm = self.beta1 * pm + (1 - self.beta1) * grad
            pv = self.beta2 * pv + (1 - self.beta2) * (grad * grad)
            mhat = pm / (1 - self.beta1 ** (it + 1))
            vhat = pv / (1 - self.beta2 ** (it + 1))
            theta += self.lr * (mhat / (np.sqrt(vhat) + self.eps))

            # Log
            training_log["iteration"].append(it + 1)
            training_log["average_feature_difference"].append(
                float(np.linalg.norm((human_feature_exp - feature_exp) / num_demo_agents))
            )
            training_log["average_log-likelihood"].append(float(np.mean(log_like_list)) if log_like_list else float("nan"))
            training_log["average_human_likeness"].append(
                float(np.mean(iteration_human_likeness)) if iteration_human_likeness else float("nan")
            )
            training_log["theta"].append(theta.copy())

            if (it + 1) % 10 == 0:
                print(f"Iteration {it + 1}: Log-likelihood = {training_log['average_log-likelihood'][-1]:.4f}")

        self.theta = theta
        return theta, training_log

    @staticmethod
    def save_results(theta: np.ndarray, training_log: Dict[str, Any], path: str = "irl_results.pkl",
                     norm_mean: Optional[np.ndarray] = None, norm_std: Optional[np.ndarray] = None) -> None:
        with open(path, "wb") as f:
            pickle.dump({"theta": theta, "training_log": training_log, "norm_mean": norm_mean, "norm_std": norm_std}, f)
        print(f"Saved IRL results to {path}")


if __name__ == "__main__":
    # Load features from output dir in config
    feature_dir = os.path.join(default_config.output_dir, "features")
    features = MaxEntIRL.load_features(feature_dir)

    # Use feature names from config so extractor/config control the set
    irl = MaxEntIRL(feature_names=default_config.feature_names)

    theta, log = irl.fit(features)
    print(f"Final learned weights: {theta}")

    weights_output_dir = os.path.join(default_config.output_dir, "irl_weights.pkl")
    MaxEntIRL.save_results(theta, log, path=weights_output_dir, norm_mean=irl.norm_mean, norm_std=irl.norm_std)