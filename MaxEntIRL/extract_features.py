import numpy as np
import os
import torch
import random
import importlib
import pickle
from .irl_config import default_config
from .visualize_rollout_gt import visualize_guided_rollout_with_gt, visualize_trajectories_simple
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvUnifiedBuilder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
import tbsim.utils.tensor_utils as TensorUtils
            
class IRLFeatureExtractor:
    def __init__(self, eval_cfg, config=default_config):
        self.config = config
        self.env = None
        self.policy = None
        self.policy_model = None
        self._setup_from_scene_editor_config(eval_cfg)
        self._filtered_scene_cache = None  # Cache for filtered scenes
        
    def _setup_from_scene_editor_config(self, eval_cfg):
        """
        Setup environment, policy, and model from scene editor configuration
        Enhanced to properly extract algorithm config for history frames
        """
        assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"
            
        # Set global batch type
        set_global_batch_type("trajdata")
        if eval_cfg.env == "nusc":
            set_global_trajdata_batch_env("nusc_trainval")
        elif eval_cfg.env == "trajdata":
            set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
        
        # Set random seeds for reproducibility
        np.random.seed(eval_cfg.seed)
        random.seed(eval_cfg.seed)
        torch.manual_seed(eval_cfg.seed)
        torch.cuda.manual_seed(eval_cfg.seed)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create policy and rollout wrapper
        policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
        composer_class = getattr(policy_composers, eval_cfg.eval_class)
        composer = composer_class(eval_cfg, device)
        policy, exp_config = composer.get_policy()
        
        # Set global trajdata batch raster config
        set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
        
        # Extract policy_model
        policy_model = None
        print('policy', policy)
        if hasattr(policy, 'model'):
            policy_model = policy.model
        
        # Set evaluation time sampling/optimization parameters
        if eval_cfg.apply_guidance:
            if eval_cfg.eval_class in ['SceneDiffuser', 'Diffuser', 'TrafficSim', 'BC', 'HierarchicalSampleNew']:
                if policy_model is not None:
                    policy_model.set_guidance_optimization_params(eval_cfg.guidance_optimization_params)
            if eval_cfg.eval_class in ['SceneDiffuser', 'Diffuser']:
                if policy_model is not None:
                    policy_model.set_diffusion_specific_params(eval_cfg.diffusion_specific_params)
        
        # Create environment
        if eval_cfg.env == "nusc":
            env_builder = EnvNuscBuilder(eval_cfg, exp_config, device)
        elif eval_cfg.env == "trajdata":
            env_builder = EnvUnifiedBuilder(eval_cfg, exp_config, device)
        else:
            raise ValueError(f"Unknown environment: {eval_cfg.env}")
        
        env = env_builder.get_env()
        
        # Store exp_config in env for access to algorithm parameters
        env.exp_config = exp_config
        env.eval_cfg = eval_cfg
        
        self.env = env
        self.policy = policy
        self.policy_model = policy_model


    def extract_irl_features_from_all_frames(self):
        """
        Extract IRL features by generating rollouts from ALL frames, not just start frame
        """
        print("Extracting IRL features from ALL frames...")
            
        all_features = {}
        scene_i = 0
        if self._filtered_scene_cache is not None:
            eval_scenes = self._filtered_scene_cache
        else:
            eval_scenes = self.filtering_scenes()
            self.config.eval_scenes = eval_scenes

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
      
            # Determine start frames for each scene
            start_frame_index: list[list[int]] = []
            valid_scene_indices = []
            valid_scene_wrappers = [] 
            
            # History frames needed by policy
            history_frames = getattr(self.env.exp_config.algo, "history_num_frames", 0)

            # Multiple sims per scene: spread starts across valid range
            for si_idx, scene_idx in enumerate(scene_indices):
                current_scene = self.env._current_scenes[si_idx].scene
                sframe = history_frames + 1
                # Ensure there's enough horizon for rollout
                eframe = current_scene.length_timesteps - self.config.horizon
                if eframe <= sframe:
                    print(f"Scene {scene_idx}: insufficient length (length={current_scene.length_timesteps}, need at least {sframe + self.config.horizon})")
                    continue
                valid_scene_indices.append(scene_idx)
                valid_scene_wrappers.append(self.env._current_scenes[si_idx]) 

                if self.config.num_sim_per_scene > 1:
                    # Multiple simulations per scene - spread them across the scene
                    scene_frame_inds = np.linspace(sframe, eframe, num=self.config.num_sim_per_scene, dtype=int).tolist()
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
            assert len(scene_indices) == len(start_frame_index), f"Mismatch: {len(scene_indices)} scenes vs {len(start_frame_index)} start_frame entries"
                    
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

                    # Capture context snapshot (map raster + ego/neighbor history)
                    # for the weight network before generating any rollouts.
                    frame_context = None
                    if getattr(self.config, "save_context", False):
                        try:
                            frame_context = self._extract_frame_context()
                        except Exception as e:
                            print(f"    Warning: failed to extract frame context: {e}")
                            frame_context = None

                    # Generate rollouts starting from this specific frame
                    rollout_trajectories, gt_trajectories = self._generate_rollouts_from_specific_frame(
                        scene_idx, start_frame)

                    if not rollout_trajectories or gt_trajectories is None:
                        print(f"    Warning: No data generated for frame {start_frame}")
                        torch.cuda.empty_cache()
                        continue

                    frame_features_data = self._process_frame_trajectories(
                        scene_idx, scene_name, start_frame, rollout_trajectories, gt_trajectories)

                    if frame_features_data:
                        entry = {
                            "start_frame": start_frame,
                            "frame_features": frame_features_data,
                        }
                        if frame_context is not None:
                            entry["context"] = frame_context
                        scene_features.append(entry)

                # return features for current scene, and save if necessary
                if scene_features:
                    all_features[f"{scene_idx}"] = scene_features
                    
                    if self.config.save_features:
                        features_output_dir = os.path.join(self.config.output_dir, "features")
                        if not os.path.exists(features_output_dir):
                            os.makedirs(features_output_dir)
                        output_path = os.path.join(features_output_dir, f"{scene_name}_irl_features.pkl")
                        with open(output_path, 'wb') as f:
                            pickle.dump(scene_features, f)

                        print(f"Saved {len(scene_features)} frame features to {output_path}")                    
                    
            torch.cuda.empty_cache()
        return all_features


    def filtering_scenes(self):
        """
        Filter scenes based on location and number of scenes to evaluate
        """ 
        valid_scene_indices = []
     
        scenes_data = list(self.env.dataset.scenes())
        all_scene_indices = list(range(len(scenes_data)))
        print(f"✅ Found {len(all_scene_indices)} total scenes in dataset")
            
        max_scenes = self.config.num_scenes_to_evaluate
        for scene_idx in all_scene_indices:
            # Check if we've reached the max_scenes limit (only if max_scenes > 0)
            if max_scenes > 0 and len(valid_scene_indices) >= max_scenes:
                break
                
            # Check scene location before adding to valid list
            scenes_valid = self.env.reset(scene_indices=[scene_idx], start_frame_index=None)
            if scenes_valid[0]:
                scene_location = self.env._current_scenes[0].scene.location
                if self.config.scene_location in scene_location:
                    valid_scene_indices.append(scene_idx)
                    print(f"  ✅ Scene {scene_idx}: {scene_location} - Added ({len(valid_scene_indices)}/{max_scenes if max_scenes > 0 else '∞'})")
                else:
                    print(f"  ❌ Scene {scene_idx}: {scene_location} - Skipped (wrong location)")
            else:
                print(f"  ❌ Scene {scene_idx}: Invalid scene - Skipped")
        
        if max_scenes > 0:
            print(f"✅ Filtered to {len(valid_scene_indices)} scenes matching location '{self.config.scene_location}' with max_scenes={max_scenes}")         
        else:
            print(f"✅ Filtered to {len(valid_scene_indices)} scenes matching location '{self.config.scene_location}' without max_scenes limit")
        
        # Cache the result for future iterations
        self._filtered_scene_cache = valid_scene_indices
        
        return valid_scene_indices

    def _generate_rollouts_from_specific_frame(self, scene_idx, start_frame):
        """
        Generate multiple rollouts starting from a SPECIFIC frame
        """
        # Step 1: Extract ground truth trajectory
        print(f"    Step 1: Extracting ground truth trajectory for scene {scene_idx}, frame {start_frame}")
        ground_truth = self._extract_ground_truth_trajectory(start_frame)

        if ground_truth is None or len(ground_truth) == 0:
            print(f"    No dynamic agents found in ground truth, skipping rollouts")
            return None, None
        
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
                        heuristic_guidance_cfg = [heuristic_guidance_cfg[i] for i in valid_scene_inds] if len(valid_scene_inds) > 0 else [[]]
                        guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)
        
        print(f"      Rollouts: guidance_config type: {type(guidance_config)}, length: {len(guidance_config) if guidance_config else 'None'}")
        print(f"      Rollouts: constraint_config type: {type(constraint_config)}, length: {len(constraint_config) if constraint_config else 'None'}")

        # Step 2: Generate rollouts
        rollouts = []
        # Generate multiple rollouts from this frame
        print(f"    Generating {self.config.num_rollouts} rollouts from frame {start_frame}")

        for rollout_idx in range(self.config.num_rollouts):
            # Reset environment to this specific frame
            scenes_valid = self.env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
            if not scenes_valid[0]:
                print(f"      Scene {scene_idx} invalid at frame {start_frame} for rollout {rollout_idx}")
                continue
            
            # Add variation for different rollouts
            if rollout_idx > 0:
                np.random.seed(42 + rollout_idx + start_frame)
                torch.manual_seed(42 + rollout_idx + start_frame)
            
            # Generate single rollout from this frame
            rollout_data = self._generate_single_rollout_from_frame(
                scene_idx, start_frame, rollout_idx,
                guidance_config=guidance_config,
                constraint_config=constraint_config,
                obs_to_torch=obs_to_torch,
                device=device
            )

            if rollout_data is not None:
                rollouts.append(rollout_data)
            else:
                print(f"      Failed to generate rollout {rollout_idx}")

        print(f"    Generated {len(rollouts)} valid rollouts out of {self.config.num_rollouts} attempts")
        # Step 3: Extract trajectories from rollouts
        rollout_trajectories = self._extract_trajectories_from_rollouts(rollouts)

        if not rollout_trajectories:
            print(f"    Warning: No dynamic agent trajectories found for frame {start_frame}")
            return [], None
            
        return rollout_trajectories, ground_truth

    def _generate_single_rollout_from_frame(self, scene_idx, start_frame, rollout_idx,
                                            guidance_config=None, constraint_config=None,
                                            obs_to_torch=None, device=None):
        """
        Generate a single rollout starting from current environment state with guidance like scene_editor.py
        """
        try:
            # Use the guided_rollout function from scene_edit_utils            
            print(f"      Rollout {rollout_idx}: Starting guided_rollout with horizon={self.config.horizon}")
            
            # Wrap policy like scene_editor.py does
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
            
            # Check if info is None before accessing it
            if info is None:
                print(f"      Rollout {rollout_idx}: WARNING - info is None!")
                return None
            
            print(f"      Rollout {rollout_idx}: guided_rollout completed successfully")
            
            # Extract buffer with proper error handling
            buffer = None
            if isinstance(info, dict) and 'buffer' in info:
                if info['buffer'] is not None and len(info['buffer']) > 0:
                    buffer = info['buffer'][0]  # First scene's buffer
                    print(f"      Rollout {rollout_idx}: Successfully extracted buffer")
                else:
                    print(f"      Rollout {rollout_idx}: Buffer is empty or None")
            else:
                print(f"      Rollout {rollout_idx}: No buffer key in info")
            
            return {
                'rollout_idx': rollout_idx,
                'stats': stats,
                'info': info,
                'buffer': buffer,
            }
            
        except Exception as e:
            print(f"      Error in rollout {rollout_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_trajectories_from_rollouts(self, rollouts):
        """
        Extract trajectory data from each rollout results
        """
        agents_trajectories_all_rollouts = []
        
        for rollout_idx, rollout in enumerate(rollouts):
            if rollout is None:
                continue
                
            buffer = rollout.get('buffer')
            if buffer is not None and isinstance(buffer, dict):
                # Extract agents trajectories from each rollout buffer
                agents_per_rollout = {}
                
                # Get agent IDs
                agent_ids = np.asarray(buffer['track_id'])                        
                unique_agent_ids = agent_ids[:, 0].tolist()  # [num_agents, timesteps], get first column for unique IDs
                
                # Get centroid data
                centroids = np.asarray(buffer['centroid'])
                yaws = np.asarray(buffer['yaw'])
                    
                # Create dictionary with agent_id as key for a rollout
                for i, unique_agent_id in enumerate(unique_agent_ids):
                    if unique_agent_id not in agents_per_rollout:
                        agents_per_rollout[unique_agent_id] = {
                            'positions': centroids[i],
                            'yaw': yaws[i]
                        }

                if agents_per_rollout:
                    agents_trajectories_all_rollouts.append(agents_per_rollout)
        print(f"      Extracted trajectories for agents from rollouts")

        return agents_trajectories_all_rollouts

    def _extract_ground_truth_trajectory(self, start_frame):
        """
        Extract ground truth from cached trajdata
        """
        try:      
            if not (hasattr(self.env, '_current_scenes') and len(self.env._current_scenes) > 0):
                return None

            current_scene = self.env._current_scenes[0]

            # Get agent names and agent objects
            if hasattr(current_scene, 'agents'):
                agents = current_scene.agents
                agent_names = [agent.name for agent in agents]
            else:
                print("      Cannot access agent names")
                return None
            
            gt_trajectories = {}
            
            # Step 1: Extract gt trajectories for each agent with full state info
            for agent_idx, (agent, agent_name) in enumerate(zip(agents, agent_names)):           
                agent_positions = []
                agent_velocities = []
                agent_accelerations = []
                agent_headings = []
                agent_speeds = []

                for frame_offset in range(self.config.horizon):
                    frame_idx = start_frame + frame_offset

                    # Check if agent is active in this frame using the agent object
                    if not (agent.first_timestep <= frame_idx < agent.last_timestep):
                        # Agent is not active in this frame, so we can safely skip it
                        continue
                        
                    # Use the dataset's method to get agent states
                    states = current_scene.cache.get_states([agent_name], frame_idx)
                    
                    # Process valid states
                    if states is not None and hasattr(states, '__len__') and len(states) > 0:
                        state = states[0]
                        # Handle tensor conversion
                        if hasattr(state, 'cpu'):
                            state_np = state.cpu().numpy()
                        elif hasattr(state, 'detach'):
                            state_np = state.detach().numpy()
                        else:
                            state_np = np.array(state)
                        
                        # Extract all state components
                        # StateArrayXYXdYdXddYddSC format: [x, y, vx, vy, ax, ay, sin_h, cos_h]
                        x, y = float(state_np[0]), float(state_np[1])
                        vx, vy = float(state_np[2]), float(state_np[3])
                        ax, ay = float(state_np[4]), float(state_np[5])
                        sin_h, cos_h = float(state_np[6]), float(state_np[7])
                        
                        # Compute derived quantities
                        speed = np.sqrt(vx**2 + vy**2)
                        heading = np.arctan2(sin_h, cos_h)  # Convert from sin/cos to angle
                        
                        # Store all data
                        agent_positions.append([x, y])
                        agent_velocities.append([vx, vy])
                        agent_accelerations.append([ax, ay])
                        agent_speeds.append(speed)
                        agent_headings.append(heading)
                    else:
                        # No data returned, skip frame
                        continue

                # Only include agents with sufficient trajectory data
                if len(agent_positions) >= 2:
                    gt_trajectories[agent_idx] = {
                        'positions': np.array(agent_positions),
                        'velocities': np.array(agent_velocities),
                        'accelerations': np.array(agent_accelerations),
                        'speeds': np.array(agent_speeds),
                        'yaw': np.array(agent_headings)
                    }                               
                else:
                    print(f"      Agent {agent_name} (ID {agent_idx}): insufficient data ({len(agent_positions)} positions)")
                    
            return gt_trajectories
            
        except Exception as e:
            print(f"      Error extracting GT from cache: {e}")
            return None

    def _process_frame_trajectories(self, scene_idx, scene_name, frame_number, rollout_trajectories, ground_truth):
        """
        Process trajectories from a specific frame and compute features with agent matching
        """
        if not rollout_trajectories or ground_truth is None:
            return None
        
        # Step 1: Filter for dynamic agents for feature computation
        dynamic_agent_ids = []
        for agent_idx, trajectory in ground_truth.items():
            if self.is_trajectory_dynamic(trajectory):
                dynamic_agent_ids.append(agent_idx)

        print(f"    Filtered to {len(dynamic_agent_ids)} dynamic agents out of {len(ground_truth)} total")

        # Compute features for all rollouts (per dynamic agent)
        dynamic_agents_features = {}
        for one_rollout in rollout_trajectories:
            agent_features = self.compute_irl_features(one_rollout, dynamic_agent_ids)
            for agent_id, features in agent_features.items():
                if agent_id not in dynamic_agents_features:
                    dynamic_agents_features[agent_id] = []
                dynamic_agents_features[agent_id].append(features)
        print(f"    Computed features for {len(dynamic_agents_features)} dynamic agents across {len(rollout_trajectories)} rollouts")

        # Compute features for ground truth (per dynamic agent)
        dynamic_gt_features = self.compute_irl_features(ground_truth, dynamic_agent_ids)

        # Add visualization at the end if debug is enabled
        if self.config.debug:        
            plot_output_dir = os.path.join(self.config.output_dir, "visualization")
            
            try:
                if not self.config.filter_dynamic:
                    dynamic_agent_ids = None  # No filtering, use all agents
                    
                visualize_guided_rollout_with_gt(
                    rollout_trajectories=rollout_trajectories,
                    ground_truth=ground_truth,
                    scene_idx=scene_idx,
                    start_frame=frame_number,
                    scene_name=scene_name,
                    output_dir=plot_output_dir,
                    dynamic_agent_ids=dynamic_agent_ids,
                )

            except Exception as e:
                print(f"    Visualization warning: {e}") 
        
        return {
            'agent_rollout_features': dynamic_agents_features,
            'agent_ground_truth_features': dynamic_gt_features
        }

    def compute_irl_features(self, agent_trajectories_dict, dynamic_agent_ids):
        """
        Compute IRL features for agent dictionary format
        Args:
            agent_trajectories_dict: Dict with agent_id as key, trajectory [time_steps, 2] as value
            dynamic_agent_ids: List of dynamic agent IDs to compute features for
        Returns:
            dict with agent_id as key, features dict as value
        """
        if not dynamic_agent_ids or not agent_trajectories_dict:
            return {}
        
        feature_names = self.config.feature_names
        dt = self.config.step_time  
        
        # Step 1: Filter to only dynamic agents that exist in trajectories
        valid_dynamic_agents = [aid for aid in dynamic_agent_ids if aid in agent_trajectories_dict]
        if not valid_dynamic_agents:
            return {}
        
        # Step 2: Get minimum trajectory length across dynamic agents only
        dynamic_lengths = [len(agent_trajectories_dict[aid]['positions']) for aid in valid_dynamic_agents]
        T = min(dynamic_lengths)
        if T < 2:
            print(f"    Insufficient trajectory length: T={T}")
            return {}

        # Step 3: Extract data arrays for dynamic agents only
        positions = []
        velocities = []
        accelerations = []
        headings = []
        speeds = []
    
        for aid in valid_dynamic_agents:
            data = agent_trajectories_dict[aid]
            # Truncate to common length T
            positions.append(data['positions'][:T])
            headings.append(data['yaw'][:T])
            
            # Use direct data if available, otherwise compute
            if 'velocities' in data and len(data['velocities']) >= T-1:
                velocities.append(data['velocities'][:T-1])
            if 'accelerations' in data and len(data['accelerations']) >= T-1:
                accelerations.append(data['accelerations'][:T-1])
            if 'speeds' in data and len(data['speeds']) >= T-1:
                speeds.append(data['speeds'][:T-1])
                
        # Step 4: Stack arrays for dynamic agents only
        pos = np.stack(positions, axis=0).astype(np.float32) # [D, T, 2]
        heading_array = np.stack(headings, axis=0).astype(np.float32)  # [D, T]
        D = pos.shape[0] # Number of dynamic agents

        # Step 5: Compute or use velocity data
        if velocities:
            vel = np.stack(velocities, axis=0).astype(np.float32)  # [D, T-1, 2]
            if speeds:
                speed = np.stack(speeds, axis=0).astype(np.float32)  # [D, T-1]
            else:
                speed = np.linalg.norm(vel, axis=2)
        else:
            # Fallback to position differences
            vel = np.diff(pos, axis=1) / dt
            speed = np.linalg.norm(vel, axis=2)

        # Step 6: Compute longitudinal acceleration and jerk
        if accelerations:
            # Direct acceleration in x,y components
            acc_xy = np.stack(accelerations, axis=0).astype(np.float32)  # [D, T-1, 2]

            # Project to longitudinal direction using velocity
            if vel.shape[1] > 0:
                # Normalize velocity vectors
                vel_norm = vel / (np.linalg.norm(vel, axis=2, keepdims=True) + 1e-8)
                # Longitudinal acceleration = dot product of acceleration with velocity direction
                a_long = np.sum(acc_xy * vel_norm, axis=2)  # [D, T-1]
            else:
                a_long = np.zeros((D, 0), dtype=np.float32)

        else:
            # Fallback: compute from speed
            if speed.shape[1] > 1:
                a_long = np.diff(speed, axis=1) / dt                
            else:
                a_long = np.zeros((D, 0), dtype=np.float32)

        # Compute jerk
        if a_long.shape[1] > 1:
            jerk = np.diff(a_long, axis=1) / dt  # [D, T-3]
        else:
            jerk = np.zeros((D, 0), dtype=np.float32)
            
        # Step 7: Compute lateral acceleration using heading data        
        if heading_array.shape[1] > 1:
            heading_diff = np.diff(heading_array, axis=1)
            # Handle angle wrapping
            heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
            heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
            yaw_rate = heading_diff / dt
        
            # Lateral acceleration = yaw_rate * speed
            min_len = min(speed.shape[1], yaw_rate.shape[1])
            if min_len > 0:
                a_lat = yaw_rate[:, :min_len] * speed[:, :min_len]
            else:
                a_lat = np.zeros((D,0),dtype=np.float32)
        else:
            a_lat = np.zeros((D,0),dtype=np.float32)

        # Step 8: Compute TIME HEADWAY for front, left, right directions
        TT = speed.shape[1]
        pos_t = pos[:, :TT, :]  # [D, TT, 2]
        speed_t = speed[:, :TT]  # [D, TT]
        heading_t = heading_array[:, :TT]  # [D, TT]
       
        # Get ALL agent trajectories for THW computation
        all_agent_positions = []
        all_agent_ids = list(agent_trajectories_dict.keys())

        for aid in all_agent_ids:
            data = agent_trajectories_dict[aid]
            # Truncate to same length as dynamic agents
            agent_pos = data['positions'][:min(T, len(data['positions']))]
            if len(agent_pos) >= TT:
                all_agent_positions.append(agent_pos[:TT])
            else:
                # Pad with last position if trajectory is shorter
                padded_pos = np.zeros((TT, 2))
                padded_pos[:len(agent_pos)] = agent_pos
                padded_pos[len(agent_pos):] = agent_pos[-1] if len(agent_pos) > 0 else [0, 0]
                all_agent_positions.append(padded_pos)
            
            
        all_pos = np.stack(all_agent_positions, axis=0).astype(np.float32)  # [A_all, TT, 2]
        
        # Initialize time headway arrays
        front_exp_thw_all = np.zeros((D, TT), dtype=np.float32)
        left_exp_thw_all = np.zeros((D, TT), dtype=np.float32)
        right_exp_thw_all = np.zeros((D, TT), dtype=np.float32)
        
        # Compute Time Headway (THW) for each dynamic agent
        for i, dyn_aid in enumerate(valid_dynamic_agents):
            dyn_idx_in_all = all_agent_ids.index(dyn_aid)
            
            for t in range(TT):
                ego_pos = pos_t[i, t]  # [2]
                ego_speed = speed_t[i, t]  # scalar
                ego_heading = heading_t[i, t]  # scalar
            
                if ego_speed < 0.1:  # If ego is nearly stationary, skip THW computation
                    continue
                
                # Compute forward and side unit vectors for ego agent
                forward_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])  # [2]
                side_vec = np.array([-np.sin(ego_heading), np.cos(ego_heading)])    # [2] (left is positive)
            
                # Track minimum THW for each direction at this timestep
                min_front_thw = float('inf')
                min_left_thw = float('inf')
                min_right_thw = float('inf')
                
                # Check all other agents
                for j, other_aid in enumerate(all_agent_ids):
                    if j == dyn_idx_in_all:  # Skip self
                        continue
                    
                    other_pos = all_pos[j, t]  # [2]
                    
                    # Vector from ego to other agent
                    rel_pos = other_pos - ego_pos  # [2]
                    distance = np.linalg.norm(rel_pos)

                    # Project relative position onto ego's coordinate system
                    longitudinal_dist = np.dot(rel_pos, forward_vec)  # Forward/backward distance
                    lateral_dist = np.dot(rel_pos, side_vec)          # Left/right distance
                    
                    # Directional classification with geometric thresholds
                    front_threshold = 1.0    # Agent must be at least 1m ahead
                    side_threshold = 0.5     # Agent must be at least 0.5m to the side
                    angle_threshold = 0.7    # ~40 degrees (cos(40°) ≈ 0.77)
                    
                    # Calculate THW for this agent
                    thw = distance / ego_speed
                    
                    # Calculate angle factor for front classification
                    if distance > 0:
                        forward_ratio = longitudinal_dist / distance
                    else:
                        forward_ratio = 0
                        
                    # Front: agent is ahead and within reasonable angle
                    if (longitudinal_dist > front_threshold and 
                        forward_ratio > angle_threshold and 
                        abs(lateral_dist) < distance * 0.6):  # Not too far to the side
                        
                        min_front_thw = min(min_front_thw, thw)
                    
                    # Left: agent is to the left side (lateral_dist > 0)
                    if lateral_dist > side_threshold:
                        min_left_thw = min(min_left_thw, thw)
                    
                    # Right: agent is to the right side (lateral_dist < 0)
                    if lateral_dist < -side_threshold:
                        min_right_thw = min(min_right_thw, thw)

                # Set minimum THW for each direction
                if min_front_thw != float('inf'):
                    front_exp_thw_all[i, t] = np.exp(-min_front_thw)
                if min_left_thw != float('inf'):
                    left_exp_thw_all[i, t] = np.exp(-min_left_thw)
                if min_right_thw != float('inf'):
                    right_exp_thw_all[i, t] = np.exp(-min_right_thw)

            # Debug info
            avg_front = np.mean(front_exp_thw_all[i])
            avg_left = np.mean(left_exp_thw_all[i])
            avg_right = np.mean(right_exp_thw_all[i])
            # print(f"    Agent {dyn_aid}: Avg THW - Front: {avg_front:.1f}s, Left: {avg_left:.1f}s, Right: {avg_right:.1f}s")

        
        # Step 9: Assemble features for each dynamic agent
        agent_features = {}
        for i, aid in enumerate(valid_dynamic_agents):  # Fix: use valid_dynamic_agents directly
            feats = {}
            
            if 'velocity' in feature_names and speed.shape[1] > 0:
                feats['velocity'] = speed[i]
            if 'a_long' in feature_names and a_long.shape[1] > 0:
                feats['a_long'] = a_long[i]
            if 'jerk_long' in feature_names and jerk.shape[1] > 0:
                feats['jerk_long'] = jerk[i]
            if 'a_lateral' in feature_names and a_lat.shape[1] > 0:
                feats['a_lateral'] = a_lat[i]
            
            # Time Headway features
            if 'front_thw' in feature_names and TT > 0:
                feats['front_thw'] = front_exp_thw_all[i]
            if 'left_thw' in feature_names and TT > 0:
                feats['left_thw'] = left_exp_thw_all[i]
            if 'right_thw' in feature_names and TT > 0:
                feats['right_thw'] = right_exp_thw_all[i]

            agent_features[aid] = feats
    
        return agent_features


    def _extract_frame_context(self):
        """
        Capture a snapshot of the current environment observation and keep the
        fields needed by WeightNetwork (map raster + ego/neighbor history) as
        cpu numpy arrays. Takes the first scene in the current env.

        Returns dict with numpy arrays (float16 for image, float32 elsewhere)
        or None if the observation cannot be obtained.
        """
        context_fields = [
            "image",
            "history_positions",
            "history_yaws",
            "history_speeds",
            "history_availabilities",
            "extent",
            "all_other_agents_history_positions",
            "all_other_agents_history_yaws",
            "all_other_agents_history_speeds",
            "all_other_agents_history_availabilities",
            "all_other_agents_extents",
        ]

        try:
            obs = self.env.get_observation()
        except Exception as e:
            print(f"    get_observation failed: {e}")
            return None

        if not isinstance(obs, dict) or "agents" not in obs:
            return None
        batch = obs["agents"]

        snapshot = {}
        for key in context_fields:
            val = None
            if isinstance(batch, dict):
                val = batch.get(key)
            else:
                val = getattr(batch, key, None)
            if val is None:
                continue
            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()
            arr = np.asarray(val)
            # Use float16 for the map raster to save disk, float32 for the rest.
            if key == "image":
                arr = arr.astype(np.float16)
            else:
                arr = arr.astype(np.float32)
            snapshot[key] = arr
        return snapshot if snapshot else None

    def is_trajectory_dynamic(self, trajectory):
        """
        Check if a trajectory represents dynamic behavior using multiple criteria
        """
        
        # Criterion 1: Maximum velocity
        max_velocity = np.max(trajectory['speeds'])
        if max_velocity > self.config.min_velocity_threshold:
            return True
        
        # Criterion 2: Average velocity
        avg_velocity = np.mean(trajectory['speeds'])
        if avg_velocity > self.config.min_velocity_threshold * 0.5:
            return True
        
        # Criterion 3: Total distance traveled
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory['positions'], axis=0), axis=1))
        if total_distance > self.config.min_distance_threshold:
            return True
        
        # Criterion 4: End-to-end displacement
        displacement = np.linalg.norm(trajectory['positions'][-1] - trajectory['positions'][0])
        if displacement > self.config.min_distance_threshold * 0.5:
            return True
        
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Use the SAME arguments as scene_editor.py
    parser.add_argument("--dataset_path", type=str, default="../behavior-generation-dataset/nuscenes",
                       help="Path to dataset")
    parser.add_argument("--registered_name", type=str, default="trajdata_nusc_diff",
                       help="Registered config name")
    parser.add_argument("--env", type=str, default="trajdata", choices=["nusc", "trajdata"],
                       help="Which env to run")
    parser.add_argument("--eval_class", type=str, default="Diffuser",
                       help="Evaluation class")

    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "none"],
        default=["config", "heuristic"],
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. If none, does not use edits."
    )

    args = parser.parse_args()
    cfg = SceneEditingConfig(registered_name=args.registered_name)

    # Override trajdata split from IRL config (default: nusc_trainval-train).
    # Must happen before the env-specific hoist loop below pops cfg.trajdata.
    if hasattr(cfg, "trajdata") and hasattr(default_config, "trajdata_source_test"):
        cfg.trajdata.trajdata_source_test = list(default_config.trajdata_source_test)
        print(f"[extract_features] Using trajdata split: {cfg.trajdata.trajdata_source_test}")

    # Set evaluation class
    if args.eval_class is not None:
        cfg.eval_class = args.eval_class
    # Set dataset path
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path   
    
    # Set environment
    if args.env is not None:
        cfg.env = args.env
    
    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]      
        
    # Copy env-specific config to global level 
    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    
    # Remove env-specific sections
    cfg.pop("nusc", None)
    cfg.pop("trajdata", None)
    
    # Set checkpoint paths   
    cfg.ckpt.policy.ckpt_dir = default_config.policy_ckpt_dir
    cfg.ckpt.policy.ckpt_key = default_config.policy_ckpt_key
    # Set results directory to match your output directory
    cfg.results_dir = default_config.output_dir
    
    try:
        # Setup environment and model
        print("Setting up environment and model...")
        extractor = IRLFeatureExtractor(cfg, default_config)
        
        # Extract features from all frames with automatic start frame determination
        print("Starting feature extraction...")
        features = extractor.extract_irl_features_from_all_frames()        
        print("Feature extraction from all frames complete!")
        print(f"Processed {len(features)} scene-start_frame combinations")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

