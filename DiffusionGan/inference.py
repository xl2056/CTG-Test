import json
import os
import pickle
from typing import Optional

import numpy as np
import torch

from MaxEntIRL.extract_features import IRLFeatureExtractor
from MaxEntIRL.irl_config import default_config
from scripts.scene_editor import dump_episode_buffer
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
import tbsim.utils.tensor_utils as TensorUtils

class AdversarialIRLDiffusionInference:

    def __init__(self, config, hdf5_dir):
        self.config = config
        self.extractor: Optional[IRLFeatureExtractor] = None
        self.env = None
        self.policy = None
        self.policy_model = None
        self.current_theta = None
        self.training_history = []
        self.theta_ema = None  # EMA of theta for stability
        self.irl_norm_mean = None
        self.irl_norm_std = None
        self.pkl_dir = f"./MaxEntIRL/irl_output/weights/{config.scene_location}_99.pkl"
        self.location_output_dir = os.path.join(hdf5_dir, config.scene_location)
        os.makedirs(self.location_output_dir, exist_ok=True)
        self.hdf5_path = os.path.join(self.location_output_dir, "data.hdf5")

        self.setup_environment()


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
        cfg.results_dir = self.location_output_dir

        # Create extractor (holds env/policy/model)
        self.extractor = IRLFeatureExtractor(cfg, self.config)
        self.env = self.extractor.env
        self.policy = self.extractor.policy
        self.policy_model = self.extractor.policy_model
        print("Environment and models setup complete")


    def convert_reward_to_guidance(self):
        """Convert learned reward weights to diffusion guidance"""

        # Define feature names to match your IRL features
        feature_names = self.config.feature_names

        # Create custom guidance based on learned reward
        reward_guidance = {
            'name': 'learned_reward_guidance',
            'weight': self.config.guidance_weight,
            'params': {
                'reward_weights': self.current_theta.tolist(),
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

    def update_diffusion_model_with_reward(self):
        """Update diffusion model using learned reward as guidance (Generator step)"""
        print("Updating diffusion model with learned reward guidance...")

        # Convert learned reward to guidance configuration
        reward_guidance = self.convert_reward_to_guidance()

        # Apply reward-based guidance to diffusion model
        self.apply_reward_guidance(reward_guidance)


    def load_pkl(self):
        with open(self.pkl_dir, 'rb') as f:
            data = pickle.load(f)
            self.current_theta = data['final_theta']
            self.irl_norm_mean = data['norm_mean']
            self.irl_norm_std  = data['norm_std']

    def run_and_save_results(self, render_to_video=True, render_to_img=False, render_cfg=None):
        scene_i = 0
        eval_scenes = self.extractor.filtering_scenes()
        result_stats = None
        
        # Set default render config if not provided
        if render_cfg is None:
            render_cfg = {
                'size': 400,
                'px_per_m': 2.0,
                'save_every_n_frames': 5,
                'draw_mode': 'action',
            }
    
        # Create visualization directory
        if render_to_video or render_to_img:
            viz_dir = os.path.join(self.location_output_dir, "viz/")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Initialize render rasterizer
            from tbsim.utils.scene_edit_utils import get_trajdata_renderer
            render_rasterizer = get_trajdata_renderer(
                self.env.eval_cfg.trajdata_source_test,
                self.env.eval_cfg.trajdata_data_dirs,
                future_sec=self.env.eval_cfg.future_sec,
                history_sec=self.env.eval_cfg.history_sec,
                raster_size=render_cfg['size'],
                px_per_m=render_cfg['px_per_m'],
                rebuild_maps=False,
                cache_location='~/.unified_data_cache'
            )
        
        while scene_i < len(eval_scenes):
            scene_indices = eval_scenes[scene_i: scene_i + self.config.num_scenes_per_batch]
            scene_i += self.config.num_scenes_per_batch
            print(f'Processing scene_indices: {scene_indices}')

            # Add check for empty scene_indices BEFORE calling reset
            if not scene_indices:
                print('No more scenes to process, breaking...')
                break

            # Reset environment to get scene information
            scenes_valid = self.env.reset(scene_indices=scene_indices, start_frame_index=None)
            scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]

            if len(scene_indices) == 0:
                print('No valid scenes in this batch, skipping...')
                torch.cuda.empty_cache()
                continue

            # Determine start frames for each scene
            start_frame_index: list[list[int]] = []
            valid_scene_indices = []
            valid_scene_wrappers = []

            # History frames needed by policy
            history_frames = getattr(self.env.exp_config.algo, "history_num_frames", 0)

            # Multiple sims per scene: spread starts across valid range
            for si_idx, scene_idx in enumerate(scene_indices):
                print(f"Processing scene {scene_idx} (index {si_idx})")

                current_scene = self.env._current_scenes[si_idx].scene
                sframe = history_frames + 1
                # Ensure there's enough horizon for rollout
                eframe = current_scene.length_timesteps - self.config.horizon
                if eframe <= sframe:
                    print(
                        f"Scene {scene_idx}: insufficient length (length={current_scene.length_timesteps}, need at least {sframe + self.config.horizon})")
                    continue
                valid_scene_indices.append(scene_idx)
                valid_scene_wrappers.append(self.env._current_scenes[si_idx])

                if self.config.num_sim_per_scene > 1:
                    # Multiple simulations per scene - spread them across the scene
                    scene_frame_inds = np.linspace(sframe, eframe, num=self.config.num_sim_per_scene,
                                                   dtype=int).tolist()
                    start_frame_index.append(scene_frame_inds)
                    print(f"Scene {scene_idx}: frames {sframe} to {eframe}, selected starts: {scene_frame_inds}")
                else:
                    # Single sim per scene: default start
                    start_frame_index.append([sframe])
                    print(f"Scene {scene_idx}: using start frame {sframe}")

            # Update scene_indices to only include valid scenes
            scene_indices = valid_scene_indices
            self.env._current_scenes = valid_scene_wrappers

            # Now scene_indices and start_frame_index have the same length
            assert len(scene_indices) == len(
                start_frame_index), f"Mismatch: {len(scene_indices)} scenes vs {len(start_frame_index)} start_frame entries"

            # Process each scene with its determined start frames
            for si_idx, scene_idx in enumerate(scene_indices):
                scene_features = []
                scene_start_frames = start_frame_index[si_idx]
                scene_name = self.env._current_scenes[si_idx].scene.name

                for start_frame in scene_start_frames:
                    print(f"\nProcessing scene {scene_idx}, start frame {start_frame}")

                    # Reset to this specific scene and start frame
                    scenes_valid = self.env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
                    if not scenes_valid[0]:
                        print(f"Scene {scene_idx} invalid at start frame {start_frame}, skipping...")
                        torch.cuda.empty_cache()
                        continue

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                    # Initialize guidance and constraint configs like scene_editor.py (lines 189-235)
                    guidance_config = None
                    constraint_config = None

                    # Get the eval_cfg to determine guidance settings
                    eval_cfg = getattr(self.env, 'eval_cfg', None)
                    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]
                    if eval_cfg is not None:
                        # Determine guidance based on editing source
                        heuristic_config = []

                        if hasattr(eval_cfg, 'edits') and hasattr(eval_cfg.edits, 'editing_source'):
                            if "heuristic" in eval_cfg.edits.editing_source:
                                # Use heuristic config
                                if eval_cfg.edits.heuristic_config is not None:
                                    heuristic_config = eval_cfg.edits.heuristic_config

                            # Getting edits from either config file or heuristics
                            if "config" in eval_cfg.edits.editing_source:
                                guidance_config = eval_cfg.edits.guidance_config
                                constraint_config = eval_cfg.edits.constraint_config

                            if "heuristic" in eval_cfg.edits.editing_source and heuristic_config is not None:
                                # Get observation for heuristic guidance computation
                                ex_obs = self.env.get_observation()

                                if obs_to_torch:
                                    ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)

                                # Compute heuristic guidance
                                heuristic_guidance_cfg = compute_heuristic_guidance(
                                    heuristic_config,
                                    self.env,
                                    [scene_idx],  # Single scene
                                    [start_frame],  # Single start frame
                                    example_batch=ex_obs['agents']
                                )
                                # Keep only scenes with at least one guidance
                                valid_scene_inds = [i for i, sc in enumerate(heuristic_guidance_cfg) if len(sc) > 0]
                                heuristic_guidance_cfg = [heuristic_guidance_cfg[i] for i in
                                                            valid_scene_inds] if len(valid_scene_inds) > 0 else [[]]
                                guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)

                    print(
                        f"      Rollouts: guidance_config type: {type(guidance_config)}, length: {len(guidance_config) if guidance_config else 'None'}")
                    print(
                        f"      Rollouts: constraint_config type: {type(constraint_config)}, length: {len(constraint_config) if constraint_config else 'None'}")

                    rollout_policy = RolloutWrapper(agents_policy=self.policy)
                    eval_cfg = getattr(self.env, 'eval_cfg', None)

                    # Call guided_rollout
                    stats, info, renderings = guided_rollout(
                        env=self.env,
                        policy=rollout_policy,
                        policy_model=self.policy_model,
                        n_step_action=eval_cfg.n_step_action,
                        guidance_config=guidance_config,  # Pass computed guidance
                        constraint_config=constraint_config,  # Pass computed constraints
                        render=False,
                        scene_indices=[scene_idx],
                        device=device,
                        obs_to_torch=obs_to_torch,
                        horizon=self.config.horizon,
                        start_frames=[start_frame],
                        eval_class='Diffuser',  # You may want to get this from eval_cfg.eval_class
                        apply_guidance=eval_cfg.apply_guidance
                    )


                    # Extract buffer with proper error handling
                    buffer = None
                    if isinstance(info, dict) and 'buffer' in info:
                        if info['buffer'] is not None and len(info['buffer']) > 0:
                            buffer = info['buffer'][0]  # First scene's buffer
                            print(f"      Successfully extracted buffer")
                        else:
                            print(f"      RolloutBuffer is empty or None")
                    else:
                        print(f"      No buffer key in info")

                    guide_agg_dict = {}
                    pop_list = []
                    for k, v in stats.items():
                        if k.split('_')[0] == 'guide':
                            guide_name = '_'.join(k.split('_')[:-1])
                            guide_scene_tag = k.split('_')[-1][:2]
                            canon_name = guide_name + '_%sg0' % (guide_scene_tag)
                            if canon_name not in guide_agg_dict:
                                guide_agg_dict[canon_name] = []
                            guide_agg_dict[canon_name].append(v)
                            # remove from stats
                            pop_list.append(k)
                    for k in pop_list:
                        stats.pop(k, None)
                    # average over all of the same guide stats in each scene
                    for k, v in guide_agg_dict.items():
                        scene_stats = np.stack(v, axis=0)  # guide_per_scenes x num_scenes (all are nan except 1)
                        stats[k] = np.mean(scene_stats, axis=0)

                    # aggregate metrics stats
                    if result_stats is None:
                        result_stats = stats
                        result_stats["scene_index"] = np.array(info["scene_index"])
                    else:
                        for k in stats:
                            if k not in result_stats:
                                result_stats[k] = stats[k]
                            else:
                                result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
                        result_stats["scene_index"] = np.concatenate(
                            [result_stats["scene_index"], np.array(info["scene_index"])])

                    # write stats to disk
                    with open(os.path.join(self.location_output_dir, "stats.json"), "w+") as fp:
                        stats_to_write = TensorUtils.map_ndarray(result_stats, lambda x: x.tolist())
                        json.dump(stats_to_write, fp)

                    
                    # Render visualization for this scene
                    if (render_to_video or render_to_img) and "buffer" in info:
                        from tbsim.utils.scene_edit_utils import visualize_guided_rollout
                        
                        # Create scene-specific directory
                        scene_viz_dir = os.path.join(viz_dir, f"scene-{scene_idx:04d}/")
                        os.makedirs(scene_viz_dir, exist_ok=True)
                        
                        # Get the buffer for this scene
                        scene_buffer = info["buffer"][0] if info["buffer"] else None
                        
                        if scene_buffer is not None:
                            # Determine guidance and constraint configs for visualization
                            viz_guidance_config = None
                            viz_constraint_config = None
                            
                            if guidance_config is not None and len(guidance_config) > 0:
                                viz_guidance_config = guidance_config[0] if isinstance(guidance_config[0], list) else guidance_config
                            if constraint_config is not None and len(constraint_config) > 0:
                                viz_constraint_config = constraint_config[0] if isinstance(constraint_config[0], list) else constraint_config
                            
                            # Visualize the rollout
                            visualize_guided_rollout(
                                scene_viz_dir,  # Scene-specific directory
                                render_rasterizer,
                                scene_name,
                                scene_buffer,
                                guidance_config=viz_guidance_config,
                                constraint_config=viz_constraint_config,
                                fps=(1.0 / self.config.step_time),
                                n_step_action=eval_cfg.n_step_action,
                                viz_diffusion_steps=False,
                                first_frame_only=render_to_img,
                                sim_num=start_frame,
                                save_every_n_frames=render_cfg['save_every_n_frames'],
                                draw_mode=render_cfg['draw_mode'],
                            )
                    
                    if "buffer" in info:
                        dump_episode_buffer(
                            info["buffer"],
                            info["scene_index"],
                            [start_frame],
                            h5_path=self.hdf5_path
                        )
                        
            torch.cuda.empty_cache()
        
        return result_stats

    def inference_adversarial(self, render_to_video=True, render_to_img=False):
        """
        Main adversarial inference loop
        Using learned reward to guide the generation of Diffusion models
        """

        # load learned reward and mean, std of features
        self.load_pkl()

        # apply learned reward as guidance
        self.update_diffusion_model_with_reward()

        # run inference with rendering
        render_cfg = {
            'size': 400,
            'px_per_m': 2.0,
            'save_every_n_frames': 5,
            'draw_mode': 'action',
        }
        
        return self.run_and_save_results(
            render_to_video=render_to_video,
            render_to_img=render_to_img,
            render_cfg=render_cfg
        )


if __name__ == "__main__":
    
    # check if output folder exists
    output_dir = "./DiffusionGan/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Run inference with rendering
    inferencer = AdversarialIRLDiffusionInference(default_config, output_dir)
    inferencer.inference_adversarial(
        render_to_video=True,  # Set to True to generate videos
        render_to_img=False    # Set to True to generate only first frame images
    )