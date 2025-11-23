import torch
import torch.nn as nn
from typing import Dict
import numpy as np
from collections import OrderedDict

from tbsim.models.base_models import RasterizedMapEncoder
from tbsim.models.diffuser_helpers import unicyle_forward_dynamics, convert_state_to_state_and_action
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils

class FlowMatchingModel(nn.Module):
    """Flow Matching model for trajectory planning based on GoalFlow."""
    
    def __init__(
        self,
        # Map encoding
        rasterized_map=True,
        use_map_feat_global=True,
        use_map_feat_grid=False,
        map_encoder_model_arch="resnet18",
        input_image_shape=None,
        map_feature_dim=256,
        map_grid_feature_dim=256,
        
        # History encoding  
        rasterized_hist=True,
        hist_num_frames=10,
        hist_feature_dim=256,
        
        # Flow matching specific
        flow_model_arch="TemporalUnet",
        horizon=16,
        observation_dim=0,
        action_dim=3,
        output_dim=3,
        
        # Training
        flow_matching_sigma=0.1,
        flow_matching_t_max=1.0,
        loss_type='l2',
        
        # Dynamics
        dynamics_type=None,
        dynamics_kwargs=None,
        
        # Normalization
        diffuser_norm_info=None,
        agent_hist_norm_info=None,
        neighbor_hist_norm_info=None,
        
        **kwargs
    ):
        super().__init__()
        
        # Store config
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.flow_matching_sigma = flow_matching_sigma
        self.flow_matching_t_max = flow_matching_t_max
        
        # Setup normalization
        self.setup_normalization(diffuser_norm_info, agent_hist_norm_info, neighbor_hist_norm_info)
        
        # Setup dynamics
        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics()
        
        # Map encoder
        if rasterized_map:
            self.map_encoder = RasterizedMapEncoder(
                model_arch=map_encoder_model_arch,
                input_image_shape=input_image_shape,
                feature_dim=map_feature_dim,
                use_spatial_softmax=False,
            )
        else:
            self.map_encoder = None
            
        # History encoders (similar to diffuser)
        # ... implement history encoding similar to diffuser.py
        
        # Flow matching network (adapt from GoalFlow)
        # This would be the core flow matching network from GoalFlow
        self.flow_net = self._create_flow_network(flow_model_arch)
        
        # Loss function
        self.loss_fn = self._create_loss_function(loss_type)
        
    def _create_dynamics(self):
        """Create dynamics model (same as diffuser)"""
        if self._dynamics_type in ["Unicycle", "UNICYCLE"]:
            from tbsim.dynamics import Unicycle
            self.dyn = Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None
            
    def _create_flow_network(self, arch):
        """Create the flow matching network architecture"""
        # This would adapt the GoalFlow network architecture
        # For now, use a simple temporal network similar to diffuser
        from tbsim.models.diffuser_helpers import TemporalUnet
        
        return TemporalUnet(
            horizon=self.horizon,
            transition_dim=self.output_dim,
            cond_dim=512,  # Will be determined by encoders
            output_dim=self.output_dim,
            dim=128,
            dim_mults=(1, 2, 4),
        )
    
    def forward(self, data_batch, num_samp=1, **kwargs):
        """Forward pass for inference"""
        aux_info = self.get_aux_info(data_batch)
        
        if kwargs.get('return_training', False):
            return self.compute_losses(data_batch)
            
        # Sample from flow matching model
        trajectories = self.sample_trajectories(data_batch, aux_info, num_samp)
        
        pred_positions = trajectories[..., :2]
        pred_yaws = trajectories[..., 2:3] if trajectories.shape[-1] > 2 else torch.zeros_like(pred_positions[..., :1])
        
        return {
            "trajectories": trajectories,
            "predictions": {
                "positions": pred_positions,
                "yaws": pred_yaws
            }
        }
    
    def sample_trajectories(self, data_batch, aux_info, num_samp=1):
        """Sample trajectories using flow matching"""
        batch_size = data_batch["target_positions"].shape[0]
        device = data_batch["target_positions"].device
        
        # Initialize with noise
        x = torch.randn(batch_size, num_samp, self.horizon, self.output_dim, device=device)
        
        # Flow matching sampling (simplified)
        num_steps = 50
        dt = self.flow_matching_t_max / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            
            # Predict velocity field
            v_pred = self.flow_net(x.flatten(0, 1), aux_info, t.repeat(num_samp))
            v_pred = v_pred.reshape(batch_size, num_samp, self.horizon, self.output_dim)
            
            # Euler step
            x = x + v_pred * dt
            
        return x
    
    def compute_losses(self, data_batch):
        """Compute flow matching loss"""
        aux_info = self.get_aux_info(data_batch)
        target_traj = self.get_target_trajectory(data_batch)
        
        batch_size = target_traj.shape[0]
        device = target_traj.device
        
        # Sample time
        t = torch.rand(batch_size, device=device) * self.flow_matching_t_max
        
        # Sample noise
        x0 = torch.randn_like(target_traj)
        
        # Linear interpolation (conditional flow matching)
        x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * target_traj
        
        # True velocity field
        v_true = target_traj - x0
        
        # Predicted velocity field
        v_pred = self.flow_net(x_t, aux_info, t)
        
        # Flow matching loss
        loss = self.loss_fn(v_pred, v_true)
        
        return OrderedDict(flow_matching_loss=loss)
        
    def get_aux_info(self, data_batch):
        """Get auxiliary information for conditioning (similar to diffuser)"""
        # Implement similar to DiffuserModel.get_aux_info
        # This would encode map features, history, etc.
        pass
        
    def get_target_trajectory(self, data_batch):
        """Extract target trajectory from data batch"""
        # Similar to diffuser's get_state_and_action_from_data_batch
        pass