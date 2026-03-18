"""
优化版：情境感知奖励模型 v2

优化内容：
1. ✓ 支持可配置的地图输入通道数（不限于3通道）
2. ✓ 添加 Dropout 防止过拟合
3. ✓ 改进轨迹编码器：添加注意力机制
4. ✓ 添加残差连接
5. ✓ 权重初始化优化
6. ✓ 添加模型配置类，便于调参
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # 地图编码器
    map_channels: int = 5           # 地图输入通道数
    map_output_dim: int = 256       # 地图特征维度
    
    # 轨迹编码器
    traj_input_dim: int = 4         # [x, y, vx, vy]
    traj_hidden_dim: int = 64
    traj_output_dim: int = 128
    traj_num_layers: int = 2
    
    # 上下文融合
    context_dim: int = 256
    
    # 权重网络
    weight_hidden_dim: int = 128
    num_features: int = 7           # 输出权重维度
    
    # 正则化
    dropout: float = 0.1
    
    # 预训练
    pretrained_resnet: bool = True


class MapEncoderV2(nn.Module):
    """
    改进的地图编码器
    - 支持任意通道数输入
    - 添加 Dropout
    """
    
    def __init__(self, input_channels=5, output_dim=256, pretrained=True, dropout=0.1):
        super().__init__()
        
        # 加载预训练 ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 修改第一层卷积以支持任意通道数
        if input_channels != 3:
            self.first_conv = nn.Conv2d(
                input_channels, 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
            # 如果是预训练模型，用原始权重的均值初始化新通道
            if pretrained:
                with torch.no_grad():
                    # 原始权重 [64, 3, 7, 7]
                    orig_weights = resnet.conv1.weight
                    # 扩展或截取到新通道数
                    if input_channels > 3:
                        # 复制并平均
                        new_weights = orig_weights.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
                    else:
                        new_weights = orig_weights[:, :input_channels, :, :]
                    self.first_conv.weight.copy_(new_weights)
        else:
            self.first_conv = resnet.conv1
        
        # 构建 backbone
        self.backbone = nn.Sequential(
            self.first_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, 224, 224]
        Returns:
            [batch_size, output_dim]
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return self.fc(features)


class TrajectoryEncoderV2(nn.Module):
    """
    改进的轨迹编码器
    - 双向 LSTM + 注意力聚合
    - 更好地处理变长序列和多个邻居
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # LSTM 编码单个轨迹
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 时间注意力（对单条轨迹的时间步做注意力）
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 邻居注意力（对多个邻居做注意力）
        self.neighbor_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, trajectories, mask=None):
        """
        Args:
            trajectories: [batch_size, num_agents, seq_len, 4]
            mask: [batch_size, num_agents] 有效智能体掩码
        Returns:
            [batch_size, output_dim]
        """
        batch_size, num_agents, seq_len, feat_dim = trajectories.shape
        
        # 展平处理所有轨迹
        traj_flat = trajectories.view(batch_size * num_agents, seq_len, feat_dim)
        
        # LSTM 编码
        lstm_out, _ = self.lstm(traj_flat)  # [B*N, T, hidden*2]
        
        # 时间注意力
        time_weights = self.time_attention(lstm_out)  # [B*N, T, 1]
        time_weights = F.softmax(time_weights, dim=1)
        agent_features = (lstm_out * time_weights).sum(dim=1)  # [B*N, hidden*2]
        
        # 恢复 batch 和 agent 维度
        agent_features = agent_features.view(batch_size, num_agents, -1)  # [B, N, hidden*2]
        
        # 邻居注意力聚合
        neighbor_weights = self.neighbor_attention(agent_features)  # [B, N, 1]
        
        if mask is not None:
            # 掩码无效邻居
            mask = mask.unsqueeze(-1).float()  # [B, N, 1]
            neighbor_weights = neighbor_weights.masked_fill(mask == 0, -1e9)
        
        neighbor_weights = F.softmax(neighbor_weights, dim=1)
        aggregated = (agent_features * neighbor_weights).sum(dim=1)  # [B, hidden*2]
        
        aggregated = self.dropout(aggregated)
        return self.fc(aggregated)


class ContextEncoderV2(nn.Module):
    """
    改进的上下文融合模块
    - 添加残差连接
    - 更深的融合网络
    """
    
    def __init__(self, map_dim=256, traj_dim=128, output_dim=256, dropout=0.1):
        super().__init__()
        
        input_dim = map_dim + traj_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
        # 残差投影（如果维度不匹配）
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, map_features, traj_features):
        """
        Args:
            map_features: [batch_size, map_dim]
            traj_features: [batch_size, traj_dim]
        Returns:
            [batch_size, output_dim]
        """
        combined = torch.cat([map_features, traj_features], dim=-1)
        out = self.fusion(combined)
        residual = self.residual_proj(combined)
        return out + residual


class WeightNetworkV2(nn.Module):
    """
    改进的权重生成网络
    - 更深的网络
    - 残差连接
    - 权重约束选项
    """
    
    def __init__(self, input_dim=256, hidden_dim=128, num_features=7, dropout=0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )
        
        # 初始化最后一层为小值，使初始权重接近零
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, context):
        """
        Args:
            context: [batch_size, input_dim]
        Returns:
            [batch_size, num_features]
        """
        return self.mlp(context)


class SituationAwareRewardModelV2(nn.Module):
    """
    优化版情境感知奖励模型
    
    输入：
        - map_image: [B, C, 224, 224] 多通道地图
        - neighbor_trajectories: [B, N, T, 4] 邻居轨迹
        - traj_mask: [B, N] 有效邻居掩码
    
    输出：
        - weights: [B, num_features] 动态权重
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        
        self.map_encoder = MapEncoderV2(
            input_channels=config.map_channels,
            output_dim=config.map_output_dim,
            pretrained=config.pretrained_resnet,
            dropout=config.dropout
        )
        
        self.traj_encoder = TrajectoryEncoderV2(
            input_dim=config.traj_input_dim,
            hidden_dim=config.traj_hidden_dim,
            output_dim=config.traj_output_dim,
            num_layers=config.traj_num_layers,
            dropout=config.dropout
        )
        
        self.context_encoder = ContextEncoderV2(
            map_dim=config.map_output_dim,
            traj_dim=config.traj_output_dim,
            output_dim=config.context_dim,
            dropout=config.dropout
        )
        
        self.weight_network = WeightNetworkV2(
            input_dim=config.context_dim,
            hidden_dim=config.weight_hidden_dim,
            num_features=config.num_features,
            dropout=config.dropout
        )
        
        self.num_features = config.num_features
        
    def forward(self, map_image, neighbor_trajectories, traj_mask=None):
        """
        Args:
            map_image: [batch_size, channels, 224, 224]
            neighbor_trajectories: [batch_size, num_agents, seq_len, 4]
            traj_mask: [batch_size, num_agents]
        Returns:
            weights: [batch_size, num_features]
        """
        map_feat = self.map_encoder(map_image)
        traj_feat = self.traj_encoder(neighbor_trajectories, traj_mask)
        context = self.context_encoder(map_feat, traj_feat)
        weights = self.weight_network(context)
        return weights
    
    def compute_reward(self, weights, features):
        """计算奖励 r = w(c)^T * φ(s,a)"""
        return (weights * features).sum(dim=-1)


# ============ 保持向后兼容的旧版类 ============

class MapEncoder(MapEncoderV2):
    """向后兼容：旧版 MapEncoder"""
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__(input_channels=3, output_dim=output_dim, pretrained=pretrained)


class TrajectoryEncoder(nn.Module):
    """向后兼容：旧版 TrajectoryEncoder"""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, trajectories, mask=None):
        batch_size, num_agents, seq_len, feat_dim = trajectories.shape
        traj_flat = trajectories.view(batch_size * num_agents, seq_len, feat_dim)
        lstm_out, (h_n, c_n) = self.lstm(traj_flat)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        agent_features = torch.cat([h_forward, h_backward], dim=-1)
        agent_features = agent_features.view(batch_size, num_agents, -1)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            agent_features = (agent_features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            agent_features = agent_features.mean(dim=1)
        return self.fc(agent_features)


class ContextEncoder(nn.Module):
    """向后兼容：旧版 ContextEncoder"""
    def __init__(self, map_dim=512, traj_dim=128, output_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(map_dim + traj_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    def forward(self, map_features, traj_features):
        combined = torch.cat([map_features, traj_features], dim=-1)
        return self.fusion(combined)


class WeightNetwork(nn.Module):
    """向后兼容：旧版 WeightNetwork"""
    def __init__(self, input_dim=256, hidden_dim=128, num_features=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_features)
        )
    def forward(self, context):
        return self.mlp(context)


class SituationAwareRewardModel(nn.Module):
    """向后兼容：旧版模型（3通道输入）"""
    def __init__(self, num_features=7, map_dim=512, traj_dim=128, context_dim=256):
        super().__init__()
        self.map_encoder = MapEncoder(output_dim=map_dim)
        self.traj_encoder = TrajectoryEncoder(output_dim=traj_dim)
        self.context_encoder = ContextEncoder(map_dim, traj_dim, context_dim)
        self.weight_network = WeightNetwork(context_dim, hidden_dim=128, num_features=num_features)
        self.num_features = num_features
        
    def forward(self, map_image, neighbor_trajectories, traj_mask=None):
        map_feat = self.map_encoder(map_image)
        traj_feat = self.traj_encoder(neighbor_trajectories, traj_mask)
        context = self.context_encoder(map_feat, traj_feat)
        weights = self.weight_network(context)
        return weights
    
    def compute_reward(self, weights, features):
        return (weights * features).sum(dim=-1)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试新版模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("测试 SituationAwareRewardModelV2")
    print("=" * 50)
    
    config = ModelConfig(
        map_channels=5,
        dropout=0.1
    )
    model = SituationAwareRewardModelV2(config).to(device)
    
    # 模拟输入
    batch_size = 4
    map_image = torch.randn(batch_size, 5, 224, 224).to(device)  # 5通道
    neighbor_traj = torch.randn(batch_size, 10, 20, 4).to(device)
    traj_mask = torch.ones(batch_size, 10).to(device)
    traj_mask[:, 5:] = 0  # 只有前5个邻居有效
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        weights = model(map_image, neighbor_traj, traj_mask)
    
    print(f"输入地图形状: {map_image.shape}")
    print(f"输入轨迹形状: {neighbor_traj.shape}")
    print(f"输出权重形状: {weights.shape}")
    print(f"权重示例: {weights[0].cpu().numpy().round(4)}")
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试不同输入产生不同输出
    print("\n测试上下文敏感性...")
    map_image2 = torch.randn(batch_size, 5, 224, 224).to(device)
    with torch.no_grad():
        weights2 = model(map_image2, neighbor_traj, traj_mask)
    
    diff = (weights - weights2).abs().mean().item()
    print(f"不同输入的权重差异: {diff:.6f}")
    if diff > 0.001:
        print("✓ 模型对不同输入产生不同输出")
    else:
        print("✗ 警告：模型输出差异过小")
