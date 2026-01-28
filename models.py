"""
models.py - 网络模型定义
============================================================
实现论文中使用的两类网络:
  1. PointNetRotationNet: 用于点云输入 (Wahba问题 + ShapeNet)
  2. ConvRotationNet: 用于图像输入 (KITTI + MAV)

每个网络支持三种输出表示:
  - 'quat': 输出4维 → 归一化为单位四元数
  - '6d':   输出6维 → Gram-Schmidt正交化 → 旋转矩阵
  - 'A':    输出10维 → 对称矩阵 → 特征分解 → 四元数 (本文方法)
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotations import RotationRepresentations, QCQPLayer


class PointNetRotationNet(nn.Module):
    """
    点云旋转回归网络

    用于:
      - 实验1: 合成 Wahba 问题 (论文 Section VI-A)
      - 实验2: ShapeNet 点云 (论文 Section VI-B)

    结构模仿论文中引用的 [41] (Zhou et al.) 的卷积结构:
      输入点云 → 1D卷积提取逐点特征 → 全局池化 → FC → 旋转参数

    Args:
        representation: 'quat' | '6d' | 'A'
        input_dim: 输入点的维度 (默认6: [u_i, v_i] 拼接)
    """

    def __init__(self, representation='A', input_dim=6):
        super().__init__()
        self.representation = representation

        # 确定输出维度
        if representation == 'quat':
            self.output_dim = 4
        elif representation == '6d':
            self.output_dim = 6
        elif representation == 'A':
            self.output_dim = 10
        else:
            raise ValueError(f"Unknown representation: {representation}")

        # 逐点特征提取 (1D卷积, 类似PointNet)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # 全局特征 → 旋转参数
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.output_dim)
        self.bn_fc1 = nn.BatchNorm1d(128)

        # 旋转表示工具
        self.rep = RotationRepresentations()
        if representation == 'A':
            self.qcqp = QCQPLayer()

    def forward(self, x):
        """
        Args:
            x: [B, N, input_dim] 输入点云
               对于Wahba问题: [B, N, 6] = [u_i; v_i]
        Returns:
            dict with keys:
                'R': [B, 3, 3] 预测旋转矩阵
                'q': [B, 4] 预测四元数 (if applicable)
                'eigenvalues': [B, 4] (only for 'A' representation)
        """
        # [B, N, C] → [B, C, N] for Conv1d
        x = x.transpose(1, 2)

        # 逐点特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局最大池化
        x = torch.max(x, dim=2)[0]  # [B, 256]

        # FC层
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)  # [B, output_dim]

        # 根据表示类型转换为旋转
        result = {}
        if self.representation == 'quat':
            q = self.rep.normalize_quaternion(x)
            R = self.rep.quaternion_to_rotation_matrix(q)
            result['q'] = q
            result['R'] = R

        elif self.representation == '6d':
            R = self.rep.sixd_to_rotation_matrix(x)
            result['R'] = R

        elif self.representation == 'A':
            q, A, eigenvalues = self.qcqp(x)
            R = self.rep.quaternion_to_rotation_matrix(q)
            result['q'] = q
            result['R'] = R
            result['A'] = A
            result['eigenvalues'] = eigenvalues

        return result


class ConvRotationNet(nn.Module):
    """
    图像旋转回归网络 (用于 KITTI / MAV 实验)

    论文 Section VI-C/D:
      输入: 两张连续图像 (拼接为6通道)
      输出: 相对旋转

    结构: 简单CNN backbone → FC → 旋转参数
    """

    def __init__(self, representation='A', input_channels=6):
        super().__init__()
        self.representation = representation

        if representation == 'quat':
            self.output_dim = 4
        elif representation == '6d':
            self.output_dim = 6
        elif representation == 'A':
            self.output_dim = 10
        else:
            raise ValueError(f"Unknown representation: {representation}")

        # CNN backbone (简化版, 可替换为ResNet)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        # FC Head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.output_dim),
        )

        self.rep = RotationRepresentations()
        if representation == 'A':
            self.qcqp = QCQPLayer()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入图像 (拼接的两帧)
        Returns:
            dict: 同 PointNetRotationNet
        """
        feat = self.features(x).flatten(1)  # [B, 256]
        out = self.head(feat)  # [B, output_dim]

        result = {}
        if self.representation == 'quat':
            q = self.rep.normalize_quaternion(out)
            R = self.rep.quaternion_to_rotation_matrix(q)
            result['q'] = q
            result['R'] = R

        elif self.representation == '6d':
            R = self.rep.sixd_to_rotation_matrix(out)
            result['R'] = R

        elif self.representation == 'A':
            q, A, eigenvalues = self.qcqp(out)
            R = self.rep.quaternion_to_rotation_matrix(q)
            result['q'] = q
            result['R'] = R
            result['A'] = A
            result['eigenvalues'] = eigenvalues

        return result
