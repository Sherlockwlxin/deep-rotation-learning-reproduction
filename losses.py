"""
losses.py - 损失函数
============================================================
实现论文 Section IV-B 中的旋转损失函数:
  - L_chord (chordal loss): 论文主要使用的损失
  - L_quat (quaternion loss)
  - L_ang (angular loss)

论文选择 chordal loss 的原因:
  1. 可以同时用于旋转矩阵和四元数输出
  2. 计算简单, 梯度稳定
  3. 与角度误差有明确的数学关系 (footnote 3)
============================================================
"""

import torch


def chordal_loss(R_pred, R_gt):
    """
    弦距离损失 (Chordal Distance Squared)
    论文 Equation (17):
        L_chord(R, R_gt) = ||R_gt - R||_F^2

    这是论文所有实验使用的损失函数.

    Args:
        R_pred: [B, 3, 3] 预测旋转矩阵
        R_gt: [B, 3, 3] 真值旋转矩阵
    Returns:
        loss: 标量, 平均弦距离损失
    """
    diff = R_gt - R_pred
    loss = torch.sum(diff ** 2, dim=(1, 2))  # [B]
    return loss.mean()


def quaternion_loss(q_pred, q_gt):
    """
    四元数距离损失
    论文 Equation (16):
        L_quat(q, q_gt) = min(||q_gt - q||^2, ||q_gt + q||^2)

    注意: 四元数 q 和 -q 表示同一个旋转, 需要取最小值.

    Args:
        q_pred: [B, 4] 预测四元数
        q_gt: [B, 4] 真值四元数
    Returns:
        loss: 标量
    """
    # 计算两个方向的距离
    d_pos = torch.sum((q_gt - q_pred) ** 2, dim=1)  # [B]
    d_neg = torch.sum((q_gt + q_pred) ** 2, dim=1)  # [B]
    loss = torch.min(d_pos, d_neg)
    return loss.mean()


def angular_loss(R_pred, R_gt):
    """
    角度损失
    论文 Equation (18):
        L_ang(R, R_gt) = ||Log(R @ R_gt^T)||^2

    Args:
        R_pred: [B, 3, 3]
        R_gt: [B, 3, 3]
    Returns:
        loss: 标量
    """
    R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return (angle ** 2).mean()
