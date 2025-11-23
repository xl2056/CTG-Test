import numpy as np
import os
import matplotlib.pyplot as plt

def transform_points_to_raster(points, raster_from_world):
    """Transform world coordinates to raster coordinates"""
    if len(points.shape) == 2:  # [N, 2] or [N, features]
        # Ensure we only use x,y coordinates
        if points.shape[1] >= 2:
            xy_points = points[:, :2]  # Take only x,y
        else:
            print(f"Warning: points shape {points.shape} has less than 2 coordinates")
            return points
            
        # Add homogeneous coordinate
        points_h = np.concatenate([xy_points, np.ones((xy_points.shape[0], 1))], axis=1)
        # Transform points  
        raster_points = points_h @ raster_from_world.T
        return raster_points[:, :2]
    else:
        # Handle batch of trajectories
        return np.array([transform_points_to_raster(traj, raster_from_world) for traj in points])


def draw_ground_truth_trajectories(ax, ground_truth, raster_from_world, start_frame, dynamic_agent_ids=None):
    """
    Draw ground truth trajectories on the plot
    
    Args:
        ax: matplotlib axes
        ground_truth: dict with agent_id as key, trajectory as value
        raster_from_world: transformation matrix from world to raster coordinates
        start_frame: start frame index for highlighting current position
        dynamic_agent_ids: List of dynamic agent IDs to filter trajectories

    Returns:
        bool: True if any ground truth was drawn
    """
    if not ground_truth:
        return False
    
    gt_drawn = False
    # Filter trajectories based on dynamic_agent_ids
    filtered_ground_truth = {}
    if dynamic_agent_ids is not None:
        for agent_id, trajectory in ground_truth.items():
            if agent_id in dynamic_agent_ids:
                filtered_ground_truth[agent_id] = trajectory
    else:
        filtered_ground_truth = ground_truth
    
    if not filtered_ground_truth:
        return False

    colors = plt.cm.Set1(np.linspace(0, 1, len(filtered_ground_truth)))

    for i, (agent_id, trajectory_data) in enumerate(filtered_ground_truth.items()):
        if len(trajectory_data['positions']) == 0:
            continue
            
        # Ensure trajectory is numpy array
        trajectory = np.array(trajectory_data['positions'])
        if len(trajectory.shape) == 1 or trajectory.shape[0] == 0:
            continue
            
        # Transform trajectory to raster coordinates
        raster_traj = transform_points_to_raster(trajectory, raster_from_world)
        
        if len(raster_traj) > 1:
            # Draw full trajectory as dashed line
            ax.plot(raster_traj[:, 0], raster_traj[:, 1], 
                   color=colors[i], linewidth=3.0, linestyle='--', 
                   alpha=0.8, label=f'GT Agent {agent_id}')           
            
            # Mark start and end positions
            ax.scatter(raster_traj[0, 0], raster_traj[0, 1],
                      color=colors[i], s=80, marker='s', 
                      edgecolors='black', linewidth=1)
            ax.scatter(raster_traj[-1, 0], raster_traj[-1, 1],
                      color=colors[i], s=80, marker='^', 
                      edgecolors='black', linewidth=1)
            
            gt_drawn = True
    
    return gt_drawn

def draw_rollout_trajectories(ax, rollout_trajectories, raster_from_world, start_frame, dynamic_agent_ids=None):
    """
    Draw rollout trajectories on the plot
    
    Args:
        ax: matplotlib axes
        rollout_trajectories: list of rollout data with agent trajectories from extract_features.py
        raster_from_world: transformation matrix from world to raster coordinates
        start_frame: start frame index for highlighting current position
        dynamic_agent_ids: List of dynamic agent IDs to filter trajectories
    """
    if not rollout_trajectories:
        return
    
    # Use different colors for different rollouts
    rollout_colors = plt.cm.Set2(np.linspace(0, 1, len(rollout_trajectories)))
    
    for rollout_idx, rollout_data in enumerate(rollout_trajectories):        
        for agent_id, trajectory_data in rollout_data.items():
            # Skip non-dynamic agents if filtering is applied
            if dynamic_agent_ids is not None and agent_id not in dynamic_agent_ids:
                continue
            if len(trajectory_data['positions']) == 0:
                continue
                
            # Ensure trajectory is numpy array
            trajectory = np.array(trajectory_data['positions'])
            if len(trajectory.shape) == 1 or trajectory.shape[0] == 0:
                continue
                
            # Transform trajectory to raster coordinates
            raster_traj = transform_points_to_raster(trajectory, raster_from_world)
            
            if len(raster_traj) > 1:
                # Draw trajectory as solid line with transparency
                alpha = 0.5 if len(rollout_trajectories) > 1 else 0.8
                ax.plot(raster_traj[:, 0], raster_traj[:, 1], 
                       color=rollout_colors[rollout_idx], linewidth=2.0, 
                       alpha=alpha, 
                       label=f'Rollout {rollout_idx} Agent {agent_id}')             

def rasterize_rendering():
    trajdata_data_dirs = {"nusc_trainval" : "../behavior-generation-dataset/nuscenes"} # "nusc_mini"
    trajdata_source_test=["nusc_trainval-val"] # "nusc_mini"
    render_cfg = {
        'size' : 400,
        'px_per_m' : 2.0,
    }
    
    from tbsim.utils.scene_edit_utils import get_trajdata_renderer
    # initialize rasterizer once for all scenes
    render_rasterizer = get_trajdata_renderer(trajdata_source_test,
                                                trajdata_data_dirs,
                                                future_sec=5.2,
                                                history_sec=3.0,
                                                raster_size=render_cfg['size'],
                                                px_per_m=render_cfg['px_per_m'],
                                                rebuild_maps=False,
                                                cache_location='~/.unified_data_cache')
    return render_rasterizer

def visualize_guided_rollout_with_gt(rollout_trajectories, ground_truth, scene_idx, 
                                   start_frame, scene_name, output_dir, dynamic_agent_ids):
    """
    Create visualization with rasterized map background using data from extract_features.py
    
    Args:
        rollout_trajectories: List of rollout data from extract_features.py
        ground_truth: Ground truth trajectories dict from extract_features.py  
        scene_idx: Scene index
        start_frame: Starting frame
        scene_name: Scene name
        output_dir: Output directory
        dynamic_agent_ids: List of dynamic agent IDs to filter trajectories
    """
    try:
        rasterizer = rasterize_rendering()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate center position from ground truth or rollouts
        all_positions = []
        
        if ground_truth:
            for agent_id, traj in ground_truth.items():
                if dynamic_agent_ids and agent_id not in dynamic_agent_ids:
                    continue  # Skip non-dynamic agents if filtering is applied
                if len(traj['positions']) > 0:
                    traj = np.array(traj['positions'])
                    if len(traj.shape) == 2 and traj.shape[1] >= 2:
                        all_positions.extend(traj[:, :2])
        
        if rollout_trajectories:
            for one_rollout in rollout_trajectories:
                for agent_id, traj in one_rollout.items():
                    if dynamic_agent_ids and agent_id not in dynamic_agent_ids:
                        continue  # Skip non-dynamic agents if filtering is applied
                    if len(traj['positions']) > 0:
                        traj = np.array(traj['positions'])
                        if len(traj.shape) == 2 and traj.shape[1] >= 2:
                            all_positions.extend(traj[:, :2])
        
        if not all_positions:
            print("No trajectory data to visualize")
            return None
        
        # Calculate center position
        all_positions = np.array(all_positions)
        ras_pos = np.mean(all_positions, axis=0)[:2]  # Ensure only x,y
        
        # Get rasterized map
        render_result = rasterizer.render(
            ras_pos=ras_pos,
            ras_yaw=0,
            scene_name=scene_name
        )
        
        state_im, raster_from_world = render_result
        # Create figure and display rasterized map
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(state_im, origin='lower', alpha=0.8)
        
        gt_drawn = False
        
        # Draw ground truth trajectories
        if ground_truth:
            gt_drawn = draw_ground_truth_trajectories(ax, ground_truth, raster_from_world, start_frame, dynamic_agent_ids)
        
        # Draw rollout trajectories  
        if rollout_trajectories:
            draw_rollout_trajectories(ax, rollout_trajectories, raster_from_world, start_frame, dynamic_agent_ids)

        # Add legend and formatting
        if gt_drawn or rollout_trajectories:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        ax.set_title(f'{scene_name} Frame {start_frame}: Rollouts vs Ground Truth (Rasterized)')
        ax.set_xlabel('Raster X')
        ax.set_ylabel('Raster Y')
        
        # Save plot
        output_path = os.path.join(output_dir, f"{scene_name}_frame_{start_frame}_rasterized.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        
        print(f"Rasterized trajectory visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        return None

# Add a convenience function for testing without rasterizer
def visualize_trajectories_simple(rollout_trajectories, ground_truth, scene_idx, 
                                start_frame, scene_name, output_dir, dynamic_agent_ids):
    """
    Simple visualization without rasterized background for testing
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw ground truth trajectories
    if ground_truth:
        gt_colors = plt.cm.Set1(np.linspace(0, 1, len(ground_truth)))
        for i, (agent_id, trajectory_data) in enumerate(ground_truth.items()):
            trajectory = np.array(trajectory_data['positions'])
            if len(trajectory) > 1 and len(trajectory.shape) == 2:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 
                       color=gt_colors[i], linewidth=3.0, linestyle='--', 
                       alpha=0.8, label=f'GT Agent {agent_id}')
                ax.scatter(trajectory[0, 0], trajectory[0, 1],
                          color=gt_colors[i], s=100, marker='s')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                          color=gt_colors[i], s=100, marker='^')
    
    # Draw rollout trajectories
    if rollout_trajectories:
        rollout_colors = plt.cm.Set2(np.linspace(0, 1, len(rollout_trajectories)))
        for rollout_idx, rollout_data in enumerate(rollout_trajectories):
            agent_trajectories = rollout_data
            for agent_id, trajectory_data in agent_trajectories.items():
                # Filter by dynamic agents if specified
                if dynamic_agent_ids is not None and agent_id not in dynamic_agent_ids:
                    continue
                trajectory = np.array(trajectory_data['positions'])
                if len(trajectory) > 1 and len(trajectory.shape) == 2:
                    alpha = 0.5 if len(rollout_trajectories) > 1 else 0.8
                    ax.plot(trajectory[:, 0], trajectory[:, 1], 
                           color=rollout_colors[rollout_idx], linewidth=2.0, 
                           alpha=alpha, 
                           label=f'Rollout {rollout_idx} Agent {agent_id}')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Scene {scene_name} Frame {start_frame}: Rollouts vs Ground Truth')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Save plot
    output_path = os.path.join(output_dir, f"scene_{scene_name}_frame_{start_frame}_simple.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Simple trajectory visualization saved to {output_path}")
    return output_path

def get_remaining_scene_length(env, start_frame):
    """
    Get the remaining length of the scene from the start frame
    """
    if hasattr(env, '_current_scenes') and len(env._current_scenes) > 0:
        current_scene = env._current_scenes[0].scene
        return current_scene.length_timesteps - start_frame
    else:
        # Fallback: estimate by stepping through
        original_frame = env._frame_index if hasattr(env, '_frame_index') else 0
        step_count = 0
        while step_count < 1000:  # Safety limit
            try:
                obs, _, done, info = env.step(None)
                step_count += 1
                if done:
                    break
            except:
                break
        return step_count

