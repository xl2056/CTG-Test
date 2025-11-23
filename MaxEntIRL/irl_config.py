import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class FeatureExtractionConfig:
    """Configuration for IRL feature extraction"""
    
    # Model and environment - Match the working scene_editor.py format
    policy_ckpt_dir: str = "./CTG/diffuser_trained_models/test/run0"  # Relative path like scene_editor
    policy_ckpt_key: str = "iter100000.ckpt"        # Full path with checkpoints/ prefix
    
    # Output settings
    output_dir: str = "./MaxEntIRL/irl_output"
    save_features: bool = False  # Whether to save extracted features

    # feature names
    feature_names: List[str] = field(default_factory=lambda: 
        ['velocity', 'a_long', 'jerk_long', 'a_lateral', 'front_thw', 'left_thw', 'right_thw'])
    
    # Scene selection
    scene_location: str = "singapore" # Options: 'boston', 'singapore'
    num_scenes_to_evaluate: int = 10
    
    eval_scenes: List[int] = field(default_factory=list)
    num_scenes_per_batch: int = 1
    num_sim_per_scene: int = 1

    debug: bool = False  # Debug mode for visualization

    filter_dynamic: bool = True  # Filter for dynamic agents only
    min_velocity_threshold: float = 2.0 # Minimum velocity threshold for filtering agents
    min_distance_threshold: float = 5.0  # Minimum distance threshold for filtering agents

    # Feature extraction parameters
    num_rollouts: int = 8
    horizon: int = 50 # Fixed horizon for generating rollouts

    # Environment settings
    env: str = "trajdata"
    eval_class: str = "Diffuser"  
    
    # History and future settings
    history_sec: float = 3.0
    future_sec: float = 5.2
    history_num_frames: int = 20
    
    # Step settings
    step_time: float = 0.1
    n_step_action: int = 5
    
    # Random seed
    seed: int = 42
    
    # minimum remaining steps for rollouts
    min_remaining_steps: int = 50    


    ############### Parameters for adversarial training

    # Number of training iterations of DiffusionGan
    num_iterations: int = 100
    theta_ema_beta: float = 0.9
    guidance_weight: float = 1.0
    
    checkpoint_frequency: int = 20  # Save every N iterations
    
    # Wandb configuration
    use_wandb: bool = True
    wandb_project: str = "adversarial-irl-diffusion"
    wandb_entity: str = "chengwang2015"  # Your wandb username/team
    wandb_run_name: str = None  # Will be auto-generated if None
    wandb_tags: List[str] = field(default_factory=lambda: ["adversarial", "irl", "diffusion"])

    
    def __post_init__(self):
        """Initialize default values after creation"""        
        print(f"✓ Policy checkpoint dir: {self.policy_ckpt_dir}")
        print(f"✓ Policy checkpoint key: {self.policy_ckpt_key}")
        
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
# Default configuration instance
default_config = FeatureExtractionConfig()