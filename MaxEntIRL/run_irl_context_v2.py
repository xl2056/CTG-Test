"""
优化版：情境感知 IRL 训练 v2

优化内容：
1. ✓ 添加 train/val 划分（可配置比例）
2. ✓ 数据有效性检查（确保上下文不是全零）
3. ✓ 更多评估指标（NLL、Top-K准确率、权重分析）
4. ✓ 从 irl_config.py 读取配置
5. ✓ 早停机制（基于验证集）
6. ✓ 学习率调度
7. ✓ 详细的训练日志和可视化
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

from .irl_config import default_config
from .models_v2 import SituationAwareRewardModelV2, ModelConfig, count_parameters


class ContextIRLDatasetV2(Dataset):
    """
    优化版数据集
    - 支持 5 通道地图
    - 数据有效性检查
    - 统计信息
    """
    
    def __init__(self, features: List[Any], feature_names: List[str], 
                 map_channels=5, max_neighbors=10, seq_len=20,
                 verbose=True):
        self.samples = []
        self.feature_names = feature_names
        self.map_channels = map_channels
        self.max_neighbors = max_neighbors
        self.seq_len = seq_len
        
        # 统计信息
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "skipped_no_context": 0,
            "skipped_zero_map": 0,
            "avg_neighbors": [],
            "num_rollouts": []
        }
        
        # 解析特征文件
        self._parse_features(features, verbose)
        
        # 计算归一化参数
        self._compute_normalization()
        
        if verbose:
            self._print_stats()
    
    def _parse_features(self, features, verbose):
        """解析特征数据，添加有效性检查"""
        for scene_data in features:
            for frame_data in scene_data:
                frame_features = frame_data.get("frame_features", {})
                agent_rollout_features = frame_features.get("agent_rollout_features", {})
                agent_gt_features = frame_features.get("agent_ground_truth_features", {})
                agent_contexts = frame_features.get("agent_contexts", {})
                
                for agent_id, gt_feat in agent_gt_features.items():
                    self.stats["total_samples"] += 1
                    
                    # 检查 rollout 特征
                    if agent_id not in agent_rollout_features:
                        continue
                    rollout_feats = agent_rollout_features[agent_id]
                    if len(rollout_feats) == 0:
                        continue
                    
                    # 检查上下文
                    agent_context = agent_contexts.get(agent_id, None)
                    if agent_context is None:
                        self.stats["skipped_no_context"] += 1
                        continue
                    
                    # 检查上下文有效性
                    if not agent_context.get("valid", False):
                        self.stats["skipped_no_context"] += 1
                        continue
                    
                    # 检查地图是否全零
                    map_img = agent_context.get("map_image")
                    if map_img is None:
                        self.stats["skipped_zero_map"] += 1
                        continue
                    
                    map_sum = np.abs(map_img).sum()
                    if map_sum < 1.0:  # 几乎全零
                        self.stats["skipped_zero_map"] += 1
                        continue
                    
                    # 有效样本
                    sample = {
                        "gt_features": self._convert_to_vector(gt_feat),
                        "rollout_features": [self._convert_to_vector(rf) for rf in rollout_feats],
                        "context": agent_context,
                        "agent_id": agent_id
                    }
                    self.samples.append(sample)
                    
                    self.stats["valid_samples"] += 1
                    self.stats["num_rollouts"].append(len(rollout_feats))
                    self.stats["avg_neighbors"].append(agent_context.get("num_neighbors", 0))
    
    def _convert_to_vector(self, feat_dict) -> np.ndarray:
        """将特征字典转换为向量"""
        vec = []
        for name in self.feature_names:
            arr = np.asarray(feat_dict[name])
            vec.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
        return np.array(vec, dtype=np.float32)
    
    def _compute_normalization(self):
        """计算特征归一化参数"""
        all_vecs = []
        for sample in self.samples:
            all_vecs.append(sample["gt_features"])
            all_vecs.extend(sample["rollout_features"])
        
        if all_vecs:
            all_vecs = np.stack(all_vecs)
            self.norm_mean = all_vecs.mean(axis=0)
            self.norm_std = all_vecs.std(axis=0)
            self.norm_std = np.where(self.norm_std < 1e-6, 1.0, self.norm_std)
        else:
            self.norm_mean = np.zeros(len(self.feature_names))
            self.norm_std = np.ones(len(self.feature_names))
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """归一化特征向量（THW 特征不归一化）"""
        no_norm_features = {'front_thw', 'left_thw', 'right_thw'}
        normalized = vec.copy()
        
        for i, name in enumerate(self.feature_names):
            if name not in no_norm_features:
                normalized[i] = (vec[i] - self.norm_mean[i]) / (self.norm_std[i] + 1e-8)
        
        return normalized
    
    def _process_context(self, context: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理上下文数据"""
        # 地图
        map_image = np.zeros((self.map_channels, 224, 224), dtype=np.float32)
        if context.get("map_image") is not None:
            ctx_map = np.array(context["map_image"])
            # 处理通道数不匹配的情况
            if ctx_map.shape[0] == self.map_channels:
                map_image = ctx_map.astype(np.float32)
            elif ctx_map.shape[0] < self.map_channels:
                map_image[:ctx_map.shape[0]] = ctx_map.astype(np.float32)
            else:
                map_image = ctx_map[:self.map_channels].astype(np.float32)
        
        # 邻居轨迹
        neighbor_traj = np.zeros((self.max_neighbors, self.seq_len, 4), dtype=np.float32)
        traj_mask = np.zeros(self.max_neighbors, dtype=np.float32)
        
        if context.get("neighbor_trajectories") is not None:
            neighbor_trajs = context["neighbor_trajectories"]
            num_neighbors = min(len(neighbor_trajs), self.max_neighbors)
            
            for i in range(num_neighbors):
                traj = np.array(neighbor_trajs[i], dtype=np.float32)
                t_len = min(len(traj), self.seq_len)
                f_dim = min(traj.shape[1] if traj.ndim > 1 else 1, 4)
                
                if traj.ndim == 1:
                    traj = traj.reshape(-1, 1)
                
                neighbor_traj[i, :t_len, :f_dim] = traj[:t_len, :f_dim]
                traj_mask[i] = 1.0
        
        return map_image, neighbor_traj, traj_mask
    
    def _print_stats(self):
        """打印数据集统计"""
        print("\n" + "=" * 50)
        print("数据集统计")
        print("=" * 50)
        print(f"总样本数: {self.stats['total_samples']}")
        print(f"有效样本数: {self.stats['valid_samples']}")
        print(f"跳过（无上下文）: {self.stats['skipped_no_context']}")
        print(f"跳过（零地图）: {self.stats['skipped_zero_map']}")
        if self.stats['avg_neighbors']:
            print(f"平均邻居数: {np.mean(self.stats['avg_neighbors']):.2f}")
        if self.stats['num_rollouts']:
            print(f"平均 rollout 数: {np.mean(self.stats['num_rollouts']):.2f}")
        print("=" * 50)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 归一化特征
        gt_feat = self._normalize(sample["gt_features"])
        rollout_feats = np.stack([self._normalize(rf) for rf in sample["rollout_features"]])
        
        # 处理上下文
        map_img, neighbor_traj, traj_mask = self._process_context(sample["context"])
        
        return {
            "gt_features": torch.tensor(gt_feat, dtype=torch.float32),
            "rollout_features": torch.tensor(rollout_feats, dtype=torch.float32),
            "map_image": torch.tensor(map_img, dtype=torch.float32),
            "neighbor_trajectories": torch.tensor(neighbor_traj, dtype=torch.float32),
            "traj_mask": torch.tensor(traj_mask, dtype=torch.float32)
        }


class SituationAwareIRLV2:
    """优化版 IRL 训练器"""
    
    def __init__(self, feature_names: List[str], model_config: ModelConfig = None,
                 lr=1e-4, weight_decay=1e-4, device=None):
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型配置
        if model_config is None:
            model_config = ModelConfig(num_features=self.num_features)
        
        # 创建模型
        self.model = SituationAwareRewardModelV2(model_config).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print(f"模型参数量: {count_parameters(self.model):,}")
        print(f"设备: {self.device}")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        计算 MaxEnt IRL 损失及相关指标
        """
        gt_features = batch["gt_features"].to(self.device)
        rollout_features = batch["rollout_features"].to(self.device)
        map_image = batch["map_image"].to(self.device)
        neighbor_traj = batch["neighbor_trajectories"].to(self.device)
        traj_mask = batch["traj_mask"].to(self.device)
        
        batch_size = gt_features.shape[0]
        num_rollouts = rollout_features.shape[1]
        
        # 生成动态权重
        weights = self.model(map_image, neighbor_traj, traj_mask)
        
        # 计算奖励
        gt_rewards = (weights * gt_features).sum(dim=-1)
        all_features = torch.cat([rollout_features, gt_features.unsqueeze(1)], dim=1)
        weights_expanded = weights.unsqueeze(1)
        all_rewards = (weights_expanded * all_features).sum(dim=-1)
        
        # MaxEnt IRL 损失
        log_partition = torch.logsumexp(all_rewards, dim=1)
        log_likelihood = gt_rewards - log_partition
        loss = -log_likelihood.mean()
        
        # 计算指标
        with torch.no_grad():
            probs = torch.softmax(all_rewards, dim=1)
            expert_prob = probs[:, -1].mean().item()
            
            # Top-K 准确率
            top1_idx = probs.argmax(dim=1)
            top1_acc = (top1_idx == num_rollouts).float().mean().item()
            
            _, top3_idx = probs.topk(3, dim=1)
            top3_acc = (top3_idx == num_rollouts).any(dim=1).float().mean().item()
        
        metrics = {
            "expert_prob": expert_prob,
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "weights": weights.detach().cpu().numpy()
        }
        
        return loss, metrics
    
    def fit(self, features: List[Any], n_epochs=100, batch_size=16,
            val_split=0.2, log_interval=10, early_stop_patience=20) -> Dict[str, Any]:
        """训练模型"""
        
        # 创建数据集
        full_dataset = ContextIRLDatasetV2(
            features, 
            self.feature_names,
            map_channels=self.model.config.map_channels
        )
        
        if len(full_dataset) == 0:
            print("错误: 没有有效的训练样本!")
            print("请检查特征文件是否包含有效的上下文数据")
            return {}
        
        # 划分训练集和验证集
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n训练配置:")
        print(f"  训练样本: {train_size}")
        print(f"  验证样本: {val_size}")
        print(f"  批次大小: {batch_size}")
        print(f"  训练轮数: {n_epochs}")
        
        # 训练日志
        training_log = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_expert_prob": [], "val_expert_prob": [],
            "train_top1_acc": [], "val_top1_acc": [],
            "train_top3_acc": [], "val_top3_acc": [],
            "weight_mean": [], "weight_std": [], "weight_variance": [],
            "lr": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # ===== 训练 =====
            self.model.train()
            train_metrics = self._run_epoch(train_loader, train=True)
            
            # ===== 验证 =====
            self.model.eval()
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, train=False)
            
            # 学习率调度
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 权重统计
            all_weights = train_metrics["all_weights"]
            weight_mean = np.mean(all_weights, axis=0)
            weight_std = np.std(all_weights, axis=0)
            weight_variance = np.var(all_weights, axis=0).mean()
            
            # 记录日志
            training_log["epoch"].append(epoch + 1)
            training_log["train_loss"].append(train_metrics["loss"])
            training_log["val_loss"].append(val_metrics["loss"])
            training_log["train_expert_prob"].append(train_metrics["expert_prob"])
            training_log["val_expert_prob"].append(val_metrics["expert_prob"])
            training_log["train_top1_acc"].append(train_metrics["top1_acc"])
            training_log["val_top1_acc"].append(val_metrics["top1_acc"])
            training_log["train_top3_acc"].append(train_metrics["top3_acc"])
            training_log["val_top3_acc"].append(val_metrics["top3_acc"])
            training_log["weight_mean"].append(weight_mean.tolist())
            training_log["weight_std"].append(weight_std.tolist())
            training_log["weight_variance"].append(float(weight_variance))
            training_log["lr"].append(current_lr)
            
            # 打印日志
            if (epoch + 1) % log_interval == 0:
                print(f"\nEpoch {epoch+1}/{n_epochs}:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Train Expert Prob: {train_metrics['expert_prob']:.4f}, Val: {val_metrics['expert_prob']:.4f}")
                print(f"  Train Top1 Acc: {train_metrics['top1_acc']:.4f}, Val: {val_metrics['top1_acc']:.4f}")
                print(f"  Weight Variance: {weight_variance:.6f}, LR: {current_lr:.2e}")
            
            # 早停检查
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\n早停: {early_stop_patience} 轮无改善")
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 保存归一化参数
        self.norm_mean = full_dataset.norm_mean
        self.norm_std = full_dataset.norm_std
        
        return training_log
    
    def _run_epoch(self, dataloader, train=True) -> Dict:
        """运行一个 epoch"""
        total_loss = 0.0
        total_expert_prob = 0.0
        total_top1_acc = 0.0
        total_top3_acc = 0.0
        all_weights = []
        num_batches = 0
        
        for batch in dataloader:
            if train:
                self.optimizer.zero_grad()
            
            loss, metrics = self.compute_loss(batch)
            
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_expert_prob += metrics["expert_prob"]
            total_top1_acc += metrics["top1_acc"]
            total_top3_acc += metrics["top3_acc"]
            all_weights.append(metrics["weights"])
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "expert_prob": total_expert_prob / num_batches,
            "top1_acc": total_top1_acc / num_batches,
            "top3_acc": total_top3_acc / num_batches,
            "all_weights": np.concatenate(all_weights, axis=0)
        }
    
    def save(self, path: str, training_log: Dict = None):
        """保存模型"""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config,
            "feature_names": self.feature_names,
            "training_log": training_log,
            "norm_mean": getattr(self, 'norm_mean', None),
            "norm_std": getattr(self, 'norm_std', None)
        }
        torch.save(save_dict, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"模型已从 {path} 加载")
        return checkpoint.get("training_log")


def load_features(feature_dir: str) -> List[Any]:
    """加载特征文件"""
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"特征目录不存在: {feature_dir}")
    
    features = []
    for filename in sorted(os.listdir(feature_dir)):
        if filename.endswith(".pkl"):
            path = os.path.join(feature_dir, filename)
            with open(path, "rb") as f:
                features.append(pickle.load(f))
    
    print(f"加载了 {len(features)} 个特征文件")
    return features


def plot_training_curves(training_log: Dict, save_path: str):
    """绘制详细的训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = training_log["epoch"]
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, training_log["train_loss"], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, training_log["val_loss"], 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (NLL)')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Expert Probability
    ax = axes[0, 1]
    ax.plot(epochs, training_log["train_expert_prob"], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, training_log["val_expert_prob"], 'r--', label='Val', linewidth=2)
    random_prob = 1.0 / 9  # 假设 8 rollouts + 1 expert
    ax.axhline(y=random_prob, color='gray', linestyle=':', label=f'Random ({random_prob:.3f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Expert Probability')
    ax.set_title('Expert Selection Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Top-K Accuracy
    ax = axes[0, 2]
    ax.plot(epochs, training_log["train_top1_acc"], 'b-', label='Train Top1', linewidth=2)
    ax.plot(epochs, training_log["val_top1_acc"], 'r--', label='Val Top1', linewidth=2)
    ax.plot(epochs, training_log["train_top3_acc"], 'g-', label='Train Top3', linewidth=1)
    ax.plot(epochs, training_log["val_top3_acc"], 'm--', label='Val Top3', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Top-K Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Weight Variance
    ax = axes[1, 0]
    ax.plot(epochs, training_log["weight_variance"], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Variance')
    ax.set_title('Cross-Sample Weight Variance')
    ax.grid(True, alpha=0.3)
    
    # 5. Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, training_log["lr"], 'orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 6. Final Weight Distribution
    ax = axes[1, 2]
    if training_log["weight_mean"]:
        final_mean = np.array(training_log["weight_mean"][-1])
        final_std = np.array(training_log["weight_std"][-1])
        feature_names = default_config.feature_names
        
        x = np.arange(len(feature_names))
        ax.bar(x, final_mean, yerr=final_std, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Weight Value')
        ax.set_title('Final Weight Distribution (Mean ± Std)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train situation-aware IRL model (v2)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: from config)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--feature_dir", type=str, default=None, help="Feature directory")
    parser.add_argument("--output_name", type=str, default="situation_aware_irl_v2.pt", help="Output filename")
    parser.add_argument("--map_channels", type=int, default=5, help="Number of map channels")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    args = parser.parse_args()
    
    # 使用配置文件的参数
    n_epochs = args.epochs if args.epochs is not None else default_config.num_iterations
    
    # 加载特征
    feature_dir = args.feature_dir or os.path.join(default_config.output_dir, "features")
    features = load_features(feature_dir)
    
    if not features:
        print("错误: 没有找到特征文件")
        print(f"请先运行: python -m MaxEntIRL.extract_features_context_v2")
        return
    
    print("\n" + "=" * 60)
    print("情境感知 IRL 训练 v2")
    print("=" * 60)
    print(f"特征目录: {feature_dir}")
    print(f"特征文件数: {len(features)}")
    print(f"训练轮数: {n_epochs}")
    print(f"地图通道数: {args.map_channels}")
    print("=" * 60)
    
    # 模型配置
    model_config = ModelConfig(
        map_channels=args.map_channels,
        num_features=len(default_config.feature_names),
        dropout=args.dropout
    )
    
    # 创建训练器
    trainer = SituationAwareIRLV2(
        feature_names=default_config.feature_names,
        model_config=model_config,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 训练
    print("\n开始训练...")
    training_log = trainer.fit(
        features,
        n_epochs=n_epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        log_interval=10,
        early_stop_patience=30
    )
    
    if not training_log:
        print("训练失败，没有有效数据")
        return
    
    # 保存模型
    output_path = os.path.join(default_config.output_dir, args.output_name)
    trainer.save(output_path, training_log)
    
    # 绘制训练曲线
    plot_path = os.path.join(default_config.output_dir, "training_curves_v2.png")
    plot_training_curves(training_log, plot_path)
    
    # 打印最终统计
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"最终训练损失: {training_log['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {training_log['val_loss'][-1]:.4f}")
    print(f"最终训练专家概率: {training_log['train_expert_prob'][-1]:.4f}")
    print(f"最终验证专家概率: {training_log['val_expert_prob'][-1]:.4f}")
    print(f"最终权重方差: {training_log['weight_variance'][-1]:.6f}")
    
    # 验证结论
    print("\n【结果验证】")
    if training_log['weight_variance'][-1] > 0.01:
        print(f"✓ 权重方差 ({training_log['weight_variance'][-1]:.4f}) > 0.01")
        print("  → 模型具有较强的上下文敏感性！")
    elif training_log['weight_variance'][-1] > 0.001:
        print(f"△ 权重方差 ({training_log['weight_variance'][-1]:.4f}) 在 0.001-0.01 之间")
        print("  → 模型具有一定的上下文敏感性")
    else:
        print(f"✗ 权重方差 ({training_log['weight_variance'][-1]:.6f}) < 0.001")
        print("  → 模型上下文敏感性较弱，需要更多数据或调整模型")
    
    print(f"\n模型已保存: {output_path}")
    print(f"训练曲线: {plot_path}")


if __name__ == "__main__":
    main()
