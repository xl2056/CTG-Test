"""
优化版：带上下文提取的特征提取器 v2

优化内容：
1. ✓ 坐标系变换：将全局坐标转换为 ego 局部坐标系（ego 朝向为正前方）
2. ✓ 完整轨迹历史：邻居轨迹使用所有时间步，而非仅第一帧
3. ✓ 改进地图表示：5通道（ego、他车位置、他车速度、他车历史轨迹、距离场）
4. ✓ 按距离排序邻居：优先保留最近的邻居
5. ✓ 添加数据有效性检查
"""
import numpy as np
import os
import torch
import pickle
import cv2

from .extract_features import IRLFeatureExtractor
from .irl_config import default_config
from tbsim.configs.scene_edit_config import SceneEditingConfig


class ContextAwareFeatureExtractorV2(IRLFeatureExtractor):
    """优化版：在原始特征提取基础上添加上下文数据"""
    
    def __init__(self, eval_cfg, config=default_config):
        super().__init__(eval_cfg, config)
        # 上下文配置
        self.map_size = 224
        self.map_channels = 5  # 增加到5通道
        self.max_neighbors = 10
        self.seq_len = 20  # 邻居轨迹历史长度
        self.resolution = 0.5  # 米/像素
        
        # 统计信息
        self.stats = {
            "total_contexts": 0,
            "valid_maps": 0,
            "valid_trajectories": 0,
            "avg_neighbors": []
        }
        
        print("✓ ContextAwareFeatureExtractorV2 初始化完成")
        print(f"  地图尺寸: {self.map_size}x{self.map_size}, 通道数: {self.map_channels}")
        print(f"  最大邻居数: {self.max_neighbors}, 轨迹长度: {self.seq_len}")

    def _get_ego_heading(self, agent_data):
        """获取 ego 的朝向角"""
        if 'yaw' in agent_data and len(agent_data['yaw']) > 0:
            return agent_data['yaw'][0]
        elif 'velocities' in agent_data and len(agent_data['velocities']) > 0:
            vel = agent_data['velocities'][0][:2]
            if np.linalg.norm(vel) > 0.1:
                return np.arctan2(vel[1], vel[0])
        return 0.0

    def _transform_to_ego_frame(self, points, ego_pos, ego_heading):
        """
        将全局坐标转换为 ego 局部坐标系
        
        Args:
            points: [..., 2] 全局坐标点
            ego_pos: [2] ego 全局位置
            ego_heading: float ego 朝向角（弧度）
            
        Returns:
            [..., 2] ego 局部坐标
        """
        # 平移
        translated = points - ego_pos
        
        # 旋转（使 ego 朝向为正前方，即 +x 方向）
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        
        if translated.ndim == 1:
            x_local = translated[0] * cos_h - translated[1] * sin_h
            y_local = translated[0] * sin_h + translated[1] * cos_h
            return np.array([x_local, y_local])
        else:
            x_local = translated[..., 0] * cos_h - translated[..., 1] * sin_h
            y_local = translated[..., 0] * sin_h + translated[..., 1] * cos_h
            return np.stack([x_local, y_local], axis=-1)

    def _transform_velocity_to_ego_frame(self, velocities, ego_heading):
        """将全局速度转换为 ego 局部坐标系"""
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        
        if velocities.ndim == 1:
            vx_local = velocities[0] * cos_h - velocities[1] * sin_h
            vy_local = velocities[0] * sin_h + velocities[1] * cos_h
            return np.array([vx_local, vy_local])
        else:
            vx_local = velocities[..., 0] * cos_h - velocities[..., 1] * sin_h
            vy_local = velocities[..., 0] * sin_h + velocities[..., 1] * cos_h
            return np.stack([vx_local, vy_local], axis=-1)

    def _extract_context_for_agent(self, agent_id, gt_trajectories, all_trajectories):
        """
        提取单个智能体的上下文数据（优化版）
        
        Returns:
            dict: {
                "map_image": np.array [5, 224, 224],
                "neighbor_trajectories": list of np.array [T, 4],
                "ego_state": np.array [5] (x, y, vx, vy, heading),
                "num_neighbors": int,
                "valid": bool
            }
        """
        context = {
            "map_image": None,
            "neighbor_trajectories": None,
            "ego_state": None,
            "num_neighbors": 0,
            "valid": False
        }
        
        try:
            if agent_id not in gt_trajectories:
                return context
                
            agent_data = gt_trajectories[agent_id]
            if len(agent_data['positions']) == 0:
                return context
            
            # 1. 提取 ego 状态
            ego_pos = np.array(agent_data['positions'][0][:2])
            ego_heading = self._get_ego_heading(agent_data)
            
            if 'velocities' in agent_data and len(agent_data['velocities']) > 0:
                ego_vel = np.array(agent_data['velocities'][0][:2])
            elif 'speeds' in agent_data and 'yaw' in agent_data:
                speed = agent_data['speeds'][0] if len(agent_data['speeds']) > 0 else 0
                yaw = agent_data['yaw'][0] if len(agent_data['yaw']) > 0 else 0
                ego_vel = np.array([speed * np.cos(yaw), speed * np.sin(yaw)])
            else:
                ego_vel = np.array([0.0, 0.0])
            
            # ego 状态：[x, y, vx, vy, heading]（局部坐标系中 x=0, y=0）
            context["ego_state"] = np.array([0.0, 0.0, np.linalg.norm(ego_vel), 0.0, ego_heading], dtype=np.float32)
            
            # 2. 提取邻居轨迹（转换到 ego 坐标系）
            neighbor_data = []
            for other_id, other_data in gt_trajectories.items():
                if other_id == agent_id:
                    continue
                if len(other_data['positions']) < 2:
                    continue
                
                # 获取邻居的位置序列
                positions = np.array(other_data['positions'])[:, :2]
                
                # 计算与 ego 的距离（用于排序）
                dist_to_ego = np.linalg.norm(positions[0] - ego_pos)
                
                # 计算速度
                if 'velocities' in other_data and len(other_data['velocities']) > 0:
                    velocities = np.array(other_data['velocities'])[:, :2]
                elif 'speeds' in other_data and 'yaw' in other_data:
                    speeds = np.array(other_data['speeds'])
                    yaws = np.array(other_data['yaw'])
                    min_len = min(len(speeds), len(yaws), len(positions))
                    vx = speeds[:min_len] * np.cos(yaws[:min_len])
                    vy = speeds[:min_len] * np.sin(yaws[:min_len])
                    velocities = np.stack([vx, vy], axis=-1)
                else:
                    dt = self.config.step_time
                    vel_diff = np.diff(positions, axis=0) / dt
                    velocities = np.vstack([vel_diff, vel_diff[-1:]])
                
                # 确保长度一致
                min_len = min(len(positions), len(velocities), self.seq_len)
                positions = positions[:min_len]
                velocities = velocities[:min_len]
                
                # 转换到 ego 局部坐标系
                local_positions = self._transform_to_ego_frame(positions, ego_pos, ego_heading)
                local_velocities = self._transform_velocity_to_ego_frame(velocities, ego_heading)
                
                # 组合为 [T, 4]: x_local, y_local, vx_local, vy_local
                traj_data = np.concatenate([local_positions, local_velocities], axis=-1).astype(np.float32)
                
                neighbor_data.append({
                    "trajectory": traj_data,
                    "distance": dist_to_ego,
                    "current_pos": local_positions[0]
                })
            
            # 按距离排序，保留最近的邻居
            neighbor_data.sort(key=lambda x: x["distance"])
            neighbor_data = neighbor_data[:self.max_neighbors]
            
            if neighbor_data:
                context["neighbor_trajectories"] = [n["trajectory"] for n in neighbor_data]
                context["num_neighbors"] = len(neighbor_data)
            
            # 3. 生成地图（5通道）
            map_image = self._create_enhanced_map(
                ego_pos, ego_heading, gt_trajectories, agent_id, neighbor_data
            )
            context["map_image"] = map_image
            context["valid"] = True
            
        except Exception as e:
            print(f"      上下文提取警告 (agent {agent_id}): {e}")
        
        return context

    def _create_enhanced_map(self, ego_pos, ego_heading, gt_trajectories, ego_id, neighbor_data):
        """
        创建增强的多通道地图（5通道）
        
        通道说明：
            0: ego 位置和朝向（中心箭头）
            1: 其他车辆当前位置
            2: 其他车辆速度强度
            3: 其他车辆历史轨迹
            4: 距离场（到最近障碍物的距离）
        """
        map_image = np.zeros((self.map_channels, self.map_size, self.map_size), dtype=np.float32)
        center = self.map_size // 2
        
        # 通道0: ego（中心，带朝向箭头）
        cv2.circle(map_image[0], (center, center), 5, 1.0, -1)
        # 画朝向箭头（在 ego 坐标系中，朝向是 +x，即向右）
        arrow_len = 15
        arrow_end = (center + arrow_len, center)
        cv2.arrowedLine(map_image[0], (center, center), arrow_end, 0.8, 2, tipLength=0.3)
        
        # 处理邻居
        obstacle_points = []
        
        for neighbor in neighbor_data:
            local_pos = neighbor["current_pos"]
            traj = neighbor["trajectory"]
            
            # 转换为像素坐标
            px = int(center + local_pos[0] / self.resolution)
            py = int(center - local_pos[1] / self.resolution)  # y 轴翻转
            
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                # 通道1: 当前位置
                cv2.circle(map_image[1], (px, py), 4, 1.0, -1)
                obstacle_points.append((px, py))
                
                # 通道2: 速度强度
                if len(traj) > 0:
                    speed = np.linalg.norm(traj[0, 2:4])
                    intensity = min(speed / 15.0, 1.0)  # 归一化到 [0, 1]
                    cv2.circle(map_image[2], (px, py), 4, intensity, -1)
                
                # 通道3: 历史轨迹
                for t in range(len(traj)):
                    hist_px = int(center + traj[t, 0] / self.resolution)
                    hist_py = int(center - traj[t, 1] / self.resolution)
                    if 0 <= hist_px < self.map_size and 0 <= hist_py < self.map_size:
                        # 轨迹点强度随时间衰减
                        decay = 1.0 - (t / len(traj)) * 0.7
                        cv2.circle(map_image[3], (hist_px, hist_py), 2, decay, -1)
        
        # 通道4: 距离场
        if obstacle_points:
            # 创建二值障碍物图
            obstacle_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
            for px, py in obstacle_points:
                cv2.circle(obstacle_map, (px, py), 4, 255, -1)
            
            # 计算距离变换
            dist_transform = cv2.distanceTransform(255 - obstacle_map, cv2.DIST_L2, 5)
            # 归一化到 [0, 1]，最大距离设为 50 像素（25米）
            map_image[4] = np.clip(dist_transform / 50.0, 0, 1)
        else:
            # 没有障碍物，距离场为全1
            map_image[4] = 1.0
        
        return map_image

    def _process_frame_trajectories(self, scene_idx, scene_name, frame_number, rollout_trajectories, ground_truth):
        """重写父类方法，添加上下文提取"""
        # 调用父类方法获取原始特征
        result = super()._process_frame_trajectories(
            scene_idx, scene_name, frame_number, rollout_trajectories, ground_truth
        )
        
        if result is None:
            return None
        
        # 提取每个动态智能体的上下文
        agent_contexts = {}
        dynamic_agent_ids = list(result['agent_ground_truth_features'].keys())
        
        for agent_id in dynamic_agent_ids:
            ctx = self._extract_context_for_agent(agent_id, ground_truth, ground_truth)
            agent_contexts[agent_id] = ctx
            
            # 更新统计
            self.stats["total_contexts"] += 1
            if ctx["valid"]:
                self.stats["valid_maps"] += 1
                if ctx["num_neighbors"] > 0:
                    self.stats["valid_trajectories"] += 1
                self.stats["avg_neighbors"].append(ctx["num_neighbors"])
        
        # 统计上下文提取结果
        valid_count = sum(1 for ctx in agent_contexts.values() if ctx["valid"])
        neighbor_counts = [ctx["num_neighbors"] for ctx in agent_contexts.values() if ctx["valid"]]
        avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
        
        print(f"    上下文提取: {len(agent_contexts)} 智能体, "
              f"有效:{valid_count}, 平均邻居数:{avg_neighbors:.1f}")
        
        # 添加上下文到结果
        result['agent_contexts'] = agent_contexts
        
        return result

    def print_stats(self):
        """打印提取统计信息"""
        print("\n" + "=" * 50)
        print("上下文提取统计")
        print("=" * 50)
        print(f"总上下文数: {self.stats['total_contexts']}")
        print(f"有效地图数: {self.stats['valid_maps']}")
        print(f"有邻居轨迹数: {self.stats['valid_trajectories']}")
        if self.stats['avg_neighbors']:
            print(f"平均邻居数: {np.mean(self.stats['avg_neighbors']):.2f}")
            print(f"最大邻居数: {max(self.stats['avg_neighbors'])}")
            print(f"最小邻居数: {min(self.stats['avg_neighbors'])}")
        print("=" * 50)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features with context (v2)")
    parser.add_argument("--dataset_path", type=str, default="/root/nuscenes")
    parser.add_argument("--registered_name", type=str, default="trajdata_nusc_diff")
    parser.add_argument("--env", type=str, default="trajdata", choices=["nusc", "trajdata"])
    parser.add_argument("--eval_class", type=str, default="Diffuser")
    parser.add_argument("--editing_source", type=str, choices=["config", "heuristic", "none"],
                       default=["config", "heuristic"], nargs="+")
    parser.add_argument("--num_scenes", type=int, default=None)

    args = parser.parse_args()
    
    # 创建配置
    cfg = SceneEditingConfig(registered_name=args.registered_name)
    
    if args.eval_class is not None:
        cfg.eval_class = args.eval_class
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path   
    if args.env is not None:
        cfg.env = args.env
    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]      
        
    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    
    cfg.pop("nusc", None)
    cfg.pop("trajdata", None)
    
    cfg.ckpt.policy.ckpt_dir = default_config.policy_ckpt_dir
    cfg.ckpt.policy.ckpt_key = default_config.policy_ckpt_key
    cfg.results_dir = default_config.output_dir
    
    if args.num_scenes is not None:
        default_config.num_scenes_to_evaluate = args.num_scenes
    
    default_config.save_features = True
    
    try:
        print("=" * 60)
        print("带上下文的特征提取 v2（优化版）")
        print("=" * 60)
        print(f"数据集路径: {args.dataset_path}")
        print(f"场景数量: {default_config.num_scenes_to_evaluate}")
        print(f"每场景帧数: {default_config.num_sim_per_scene}")
        print(f"Rollout数量: {default_config.num_rollouts}")
        print(f"输出目录: {default_config.output_dir}")
        print("=" * 60)
        
        print("\n[1/2] 初始化环境和模型...")
        extractor = ContextAwareFeatureExtractorV2(cfg, default_config)
        
        print("\n[2/2] 开始特征提取...")
        features = extractor.extract_irl_features_from_all_frames()
        
        # 打印统计信息
        extractor.print_stats()
        
        print("\n" + "=" * 60)
        print("特征提取完成!")
        print(f"处理了 {len(features)} 个场景")
        print(f"特征文件保存在: {os.path.join(default_config.output_dir, 'features')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
