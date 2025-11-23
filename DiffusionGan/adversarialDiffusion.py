import numpy as np
import torch
import pickle
import os
import wandb
from MaxEntIRL.extract_features import IRLFeatureExtractor
from MaxEntIRL.run_irl import MaxEntIRL
from tbsim.configs.scene_edit_config import SceneEditingConfig
from MaxEntIRL.irl_config import default_config

class AdversarialIRLDiffusion:
    def __init__(self, config):
        self.config = config
        self.extractor: IRLFeatureExtractor | None = None
        self.env = None
        self.policy = None
        self.policy_model = None
        self.current_theta = None
        self.training_history = []
        self.theta_ema = None # EMA of theta for stability
        self.irl_norm_mean = None
        self.irl_norm_std = None
        
        # Initialize wandb
        if self.config.use_wandb:
            self.init_wandb()

    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = {
            "num_iterations": self.config.num_iterations,
            "theta_ema_beta": self.config.theta_ema_beta,
            "guidance_weight": self.config.guidance_weight,
            "step_time": self.config.step_time,
            "num_scenes_to_evaluate": self.config.num_scenes_to_evaluate,
            "num_rollouts": self.config.num_rollouts,
            "horizon": self.config.horizon,
            "feature_names": self.config.feature_names,
            "policy_ckpt_dir": self.config.policy_ckpt_dir,
            "policy_ckpt_key": self.config.policy_ckpt_key,
            "env": self.config.env,
            "eval_class": self.config.eval_class,
        }
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=wandb_config,
            reinit=True
        )
        
        print(f"Initialized wandb project: {self.config.wandb_project}")

    def setup_environment(self):
        """Setup environment and models"""
        # Build SceneEditingConfig similarly to extract_features.__main__
        cfg = SceneEditingConfig(registered_name="trajdata_nusc_diff")
        # Apply CLI-like overrides from default_config
        if hasattr(self.config, "eval_class"):
            cfg.eval_class = self.config.eval_class
        if hasattr(self.config, "env"):
            cfg.env = self.config.env
        # editing source (optional)
        if hasattr(cfg, "edits") and hasattr(cfg.edits, "editing_source"):
            if not isinstance(cfg.edits.editing_source, list):
                cfg.edits.editing_source = [cfg.edits.editing_source]
        # Hoist env-specific entries
        for k in cfg[cfg.env]:
            cfg[k] = cfg[cfg.env][k]
        cfg.pop("nusc", None)
        cfg.pop("trajdata", None)
        # Checkpoint paths and output dir
        cfg.ckpt.policy.ckpt_dir = self.config.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = self.config.policy_ckpt_key
        cfg.results_dir = self.config.output_dir

        # Create extractor (holds env/policy/model)
        self.extractor = IRLFeatureExtractor(cfg, self.config)
        self.env = self.extractor.env
        self.policy = self.extractor.policy
        self.policy_model = self.extractor.policy_model
        print("Environment and models setup complete")
    
    def train_adversarial(self, num_iterations=100):
        """
        Main adversarial training loop
        Generator (Diffusion): Tries to generate realistic trajectories
        Discriminator (MaxEnt IRL): Tries to distinguish real vs generated trajectories
        """
        for iteration in range(num_iterations):
            print(f"\n=== Adversarial Training Iteration {iteration + 1}/{num_iterations} ===")
            
            # Step 1: Generate trajectories using current diffusion model (Generator)
            generated_features = self.generate_trajectories_with_current_model()
            
            # Step 2: Update reward function using MaxEnt IRL (Discriminator)
            self.current_theta = self.get_reward_via_irl(generated_features)

            # Maintain an EMA of theta for smoother guidance
            beta = self.config.theta_ema_beta
            if self.current_theta is not None:
                if self.theta_ema is None:
                    self.theta_ema = np.array(self.current_theta, dtype=float)
                else:
                    self.theta_ema = beta * self.theta_ema + (1.0 - beta) * np.array(self.current_theta, dtype=float)

            # Step 3: Apply learned reward as guidance
            if iteration < num_iterations - 1:
                self.update_diffusion_model_with_reward()
            
            # Step 4: Evaluate and log progress
            self.evaluate_iteration(generated_features, iteration)

        return self.current_theta, self.irl_norm_mean, self.irl_norm_std
    
    def generate_trajectories_with_current_model(self):
        """Generate trajectories using current diffusion model state"""
        print("Generating trajectories with current diffusion model...")

        if self.extractor is None:
            raise RuntimeError("Extractor is not initialized. Call setup_environment() first.")
        # Extract features using current extractor (env, policy, model inside)
        features = self.extractor.extract_irl_features_from_all_frames()
        return features

    def get_reward_via_irl(self, generated_features):
        """Update reward function using MaxEnt IRL (Discriminator step)"""
        print("Updating reward function using MaxEnt IRL...")        
        
        # Run MaxEnt IRL to learn reward
        features_list = list(generated_features.values())
        irl = MaxEntIRL(feature_names=self.config.feature_names, n_iters=self.config.num_iterations)
        learned_theta, training_log = irl.fit(features_list)
        
        # Log IRL training progress to wandb
        if self.config.use_wandb:
            for i, (log_likelihood, feature_diff, human_likeness) in enumerate(zip(
                training_log["average_log-likelihood"],
                training_log["average_feature_difference"], 
                training_log["average_human_likeness"]
            )):
                wandb.log({
                    "irl/log_likelihood": log_likelihood,
                    "irl/feature_difference": feature_diff,
                    "irl/human_likeness": human_likeness,
                    "irl/iteration": i + 1
                }, commit=False)
        
        # capture normalization stats for guidance
        self.irl_norm_mean = irl.norm_mean
        self.irl_norm_std = irl.norm_std
        print(f"Learned reward weights: {learned_theta}")
        
        return learned_theta

    def update_diffusion_model_with_reward(self):
        """Update diffusion model using learned reward as guidance (Generator step)"""
        print("Updating diffusion model with learned reward guidance...")
        
        # Convert learned reward to guidance configuration
        reward_guidance = self.convert_reward_to_guidance()
        
        # Apply reward-based guidance to diffusion model
        self.apply_reward_guidance(reward_guidance)
    
    def convert_reward_to_guidance(self):
        """Convert learned reward weights to diffusion guidance"""
        if self.current_theta is None:
            return None        
        
        # Use EMA weights for stability (fallback to current if ema missing)
        theta = self.theta_ema if self.theta_ema is not None else np.array(self.current_theta, dtype=float)
        
        # Define feature names to match your IRL features
        feature_names = self.config.feature_names
               
        # Create custom guidance based on learned reward
        reward_guidance = {
            'name': 'learned_reward_guidance',
            'weight': self.config.guidance_weight,
            'params': {
                'reward_weights': theta.tolist(), 
                'feature_names': feature_names,
                'dt': self.config.step_time,
                'norm_mean': self.irl_norm_mean.tolist(),
                'norm_std': self.irl_norm_std.tolist(),
            },
            'agents': None  # Apply to all agents
        }

        return reward_guidance


    def apply_reward_guidance(self, reward_guidance):
        """Apply learned reward as guidance to diffusion model"""
        if reward_guidance is None:
            print("No reward guidance to apply")
            return
        
        # Attach to eval_cfg so extractor’s guided_rollout picks it up
        if self.env is not None and hasattr(self.env, "eval_cfg"):
            eval_cfg = self.env.eval_cfg
            if hasattr(eval_cfg, "edits"):
                eval_cfg.apply_guidance = True
                
                if not hasattr(eval_cfg.edits, "guidance_config") or eval_cfg.edits.guidance_config is None:
                    eval_cfg.edits.guidance_config = []
                # Flatten per-scene format if present
                if len(eval_cfg.edits.guidance_config) > 0 and isinstance(eval_cfg.edits.guidance_config[0], list):
                    scenes_cfg = eval_cfg.edits.guidance_config
                else:
                    scenes_cfg = [eval_cfg.edits.guidance_config]
                # Replace any previous learned_reward_guidance in each scene
                new_scenes_cfg = []
                for scene_list in scenes_cfg:
                    filtered = [g for g in scene_list if g.get('name') != 'learned_reward_guidance']
                    filtered.append(reward_guidance)
                    new_scenes_cfg.append(filtered)
                eval_cfg.edits.guidance_config = new_scenes_cfg
                print(f"Applied reward guidance with weights: {reward_guidance['params']['reward_weights']}")


    def evaluate_iteration(self, generated_features, iteration):
        """Evaluate progress and log results"""
        # Compute metrics to track training progress
        metrics = {
            'iteration': iteration,
            'reward_weights': self.current_theta.copy() if self.current_theta is not None else None,
            'reward_magnitude': np.linalg.norm(self.current_theta) if self.current_theta is not None else 0,
        }
        
        # Add trajectory quality metrics if available
        quality_metrics = self.compute_trajectory_quality_metrics(generated_features)
        metrics.update(quality_metrics)
        
        self.training_history.append(metrics)
        
        # Log to wandb
        if self.config.use_wandb:
            wandb_metrics = {
                "iteration": iteration,
                "reward/magnitude": metrics['reward_magnitude'],
                "quality/collision_rate": quality_metrics['collision_rate'],
                "quality/rollout_collision_rate": quality_metrics['rollout_collision_rate'],
                "quality/expert_similarity": quality_metrics['expert_similarity'],
                "quality/diversity": quality_metrics['diversity'],
            }
            
            # Log individual reward weights
            if self.current_theta is not None:
                for i, (weight, feature_name) in enumerate(zip(self.current_theta, self.config.feature_names)):
                    wandb_metrics[f"theta/{feature_name}"] = weight
            wandb.log(wandb_metrics)
            
        # Save checkpoint
        self.save_checkpoint(iteration)
        
        print(f"Iteration {iteration} metrics: {metrics}")

    def compute_trajectory_quality_metrics(self, generated_features):
        """
        Compute metrics directly from generated feature dictionaries (no persistent state).
        Derives:
          - collision_rate: fraction of agent-time steps with min_dis < threshold
          - rollout_collision_rate: fraction of rollouts with any collision
          - expert_similarity: mean min L2 distance (feature space) to GT across agents
          - diversity: avg pairwise L2 distance between rollout feature vectors
        """
        if not generated_features:
            return {
                'collision_rate': 0.0,
                'rollout_collision_rate': 0.0,
                'expert_similarity': 0.0,
                'diversity': 0.0,
            }

        threshold = getattr(self.config, "collision_threshold", 2.0)

        total_agent_time = 0
        total_collisions = 0
        rollouts_with_collision = 0
        total_rollouts = 0

        expert_dists = []
        diversity_dists = []

        feat_names = self.config.feature_names

        for scene_entries in generated_features.values():
            for frame_entry in scene_entries:
                ff = frame_entry.get("frame_features", {})
                agent_rollout = ff.get("agent_rollout_features", {})
                agent_gt = ff.get("agent_ground_truth_features", {})

                # Collision metrics from min_dis
                for agent_id, rollout_feat_list in agent_rollout.items():
                    if not rollout_feat_list:
                        continue
                    total_rollouts += 1
                    any_collision_this_rollout = False

                    for feat_dict in rollout_feat_list:
                        if 'min_dis' in feat_dict:
                            md = np.asarray(feat_dict['min_dis'])
                            if md.size == 0:
                                continue
                            coll_steps = int(np.sum(md < threshold))
                            total_collisions += coll_steps
                            total_agent_time += int(md.size)
                            if coll_steps > 0:
                                any_collision_this_rollout = True
                    if any_collision_this_rollout:
                        rollouts_with_collision += 1

                # Expert similarity and diversity (feature-space)
                for agent_id, rollout_feat_list in agent_rollout.items():
                    gt_feat = agent_gt.get(agent_id)
                    rollout_vecs = [self._feature_dict_to_vec(fd, feat_names) for fd in rollout_feat_list]

                    if gt_feat is not None and rollout_vecs:
                        gt_vec = self._feature_dict_to_vec(gt_feat, feat_names)
                        dists = [float(np.linalg.norm(rv - gt_vec)) for rv in rollout_vecs]
                        expert_dists.append(np.min(dists))

                    if len(rollout_vecs) >= 2:
                        R = len(rollout_vecs)
                        for i in range(R):
                            for j in range(i + 1, R):
                                diversity_dists.append(float(np.linalg.norm(rollout_vecs[i] - rollout_vecs[j])))

        collision_rate = (total_collisions / total_agent_time) if total_agent_time > 0 else 0.0
        rollout_collision_rate = (rollouts_with_collision / total_rollouts) if total_rollouts > 0 else 0.0
        expert_similarity = float(np.mean(expert_dists)) if expert_dists else 0.0
        diversity = float(np.mean(diversity_dists)) if diversity_dists else 0.0

        return {
            'collision_rate': collision_rate,
            'rollout_collision_rate': rollout_collision_rate,
            'expert_similarity': expert_similarity,
            'diversity': diversity,
        }

    def _feature_dict_to_vec(self, feat_dict, feature_names):
        """Mean-aggregate time-series features into a fixed vector in feature_names order."""
        vals = []
        for name in feature_names:
            arr = np.asarray(feat_dict.get(name, []))
            vals.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
        return np.array(vals, dtype=float)
    
    def save_checkpoint(self, iteration):
        """Save training checkpoint"""
        if iteration == self.config.num_iterations - 1:
            checkpoint = {
                'iteration': iteration,
                'theta': self.current_theta,
                'training_history': self.training_history,
            }
            
            # Create weights folder under output directory
            weights_dir = os.path.join(self.config.output_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            checkpoint_path = os.path.join(weights_dir, f"weights_{self.config.scene_location}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"Checkpoint saved to {checkpoint_path}")
        
        
        
if __name__ == "__main__":
    # Initialize adversarial trainer
    trainer = AdversarialIRLDiffusion(default_config)
    
    # Setup environment
    trainer.setup_environment()
    
    # Run adversarial training
    final_theta, norm_mean, norm_std = trainer.train_adversarial(num_iterations=default_config.num_iterations)
    
    print(f"Final learned reward weights: {final_theta}")
    
    # Save final results
    with open("adversarial_irl_results.pkl", "wb") as f:
        pickle.dump({
            "final_theta": final_theta,
            "norm_mean": norm_mean, 
            "norm_std": norm_std
        }, f)