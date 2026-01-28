"""
rotations.py - 旋转表示模块
============================================================
实现论文中三种旋转表示方法:
  1. 单位四元数 (quat) - 4维, 不连续
  2. 6D 表示 (6D) - Zhou et al. [41] 的连续表示
  3. 对称矩阵 A (ours) - 本文提出的10维表示

核心思想 (论文 Section III):
  网络输出10个参数 θ ∈ R^10, 构成 4×4 对称矩阵 A(θ).
  旋转 = A 的最小特征值对应的特征向量 (即为四元数).
  这个表示:
    - 满足光滑性 (smooth global section, Theorem 1)
    - 编码 Bingham 分布 → 自带不确定性估计
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationRepresentations:
    """三种旋转表示的转换工具类"""

    # ========================
    # 1. 对称矩阵 A (本文方法)
    # ========================

    @staticmethod
    def theta_to_A(theta):
        """
        将10维参数向量 θ 转换为 4×4 对称矩阵 A(θ)
        对应论文 Equation (8):
            A(θ) = [[θ1, θ2, θ3, θ4],
                     [θ2, θ5, θ6, θ7],
                     [θ3, θ6, θ8, θ9],
                     [θ4, θ7, θ9, θ10]]

        Args:
            theta: [B, 10] 网络输出的10维参数
        Returns:
            A: [B, 4, 4] 对称矩阵
        """
        batch_size = theta.shape[0]
        A = torch.zeros(batch_size, 4, 4, device=theta.device, dtype=theta.dtype)

        # 填充上三角 + 对角线
        A[:, 0, 0] = theta[:, 0]
        A[:, 0, 1] = theta[:, 1]
        A[:, 0, 2] = theta[:, 2]
        A[:, 0, 3] = theta[:, 3]
        A[:, 1, 1] = theta[:, 4]
        A[:, 1, 2] = theta[:, 5]
        A[:, 1, 3] = theta[:, 6]
        A[:, 2, 2] = theta[:, 7]
        A[:, 2, 3] = theta[:, 8]
        A[:, 3, 3] = theta[:, 9]

        # 对称化: 下三角 = 上三角的转置
        A = A + A.transpose(1, 2) - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        return A

    @staticmethod
    def A_to_quaternion(A):
        """
        从对称矩阵 A 提取旋转四元数 (Problem 3 的解)
        解 = A 的最小特征值对应的特征向量

        对应论文 Section III-A:
            q* = argmin_{q∈S³} q^T A q
            解为 A 的最小特征值对应的特征向量

        Args:
            A: [B, 4, 4] 对称矩阵
        Returns:
            q: [B, 4] 单位四元数
            eigenvalues: [B, 4] 所有特征值 (升序), 用于不确定性估计
        """
        # torch.linalg.eigh 返回升序排列的特征值
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        # 最小特征值对应的特征向量 = 第0列
        q = eigenvectors[:, :, 0]
        return q, eigenvalues

    @staticmethod
    def compute_dispersion(eigenvalues):
        """
        计算 Bingham 分布的色散度 tr(Λ) 作为不确定性指标
        对应论文 Equation (26):
            tr(Λ) = 3λ1 - λ2 - λ3 - λ4
        值越大(绝对值越大) → 越确定; 值越小 → 越不确定

        Args:
            eigenvalues: [B, 4] 升序排列的特征值
        Returns:
            dispersion: [B] 色散度
        """
        # eigenvalues 已经是升序: λ1 ≤ λ2 ≤ λ3 ≤ λ4
        return 3 * eigenvalues[:, 0] - eigenvalues[:, 1] - eigenvalues[:, 2] - eigenvalues[:, 3]

    # ========================
    # 2. 6D 连续表示 (Zhou et al. [41])
    # ========================

    @staticmethod
    def sixd_to_rotation_matrix(sixd):
        """
        将6D表示转换为旋转矩阵
        方法: 取前两列做 Gram-Schmidt 正交化

        Args:
            sixd: [B, 6] 6维向量
        Returns:
            R: [B, 3, 3] 旋转矩阵
        """
        a1 = sixd[:, 0:3]
        a2 = sixd[:, 3:6]

        # Gram-Schmidt 正交化
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)

        R = torch.stack([b1, b2, b3], dim=2)  # [B, 3, 3]
        return R

    # ========================
    # 3. 四元数表示
    # ========================

    @staticmethod
    def normalize_quaternion(q):
        """
        归一化四元数到单位球面

        Args:
            q: [B, 4] 四元数
        Returns:
            q_normalized: [B, 4] 单位四元数
        """
        return F.normalize(q, dim=1)

    # ========================
    # 通用转换工具
    # ========================

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        单位四元数 → 旋转矩阵
        采用 Hamilton 约定: q = (w, x, y, z), w 为标量部分

        Args:
            q: [B, 4] 单位四元数 (w, x, y, z)
        Returns:
            R: [B, 3, 3] 旋转矩阵
        """
        q = F.normalize(q, dim=1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)
        ], dim=1).reshape(-1, 3, 3)

        return R

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """
        旋转矩阵 → 单位四元数 (Shepperd's method, 数值稳定)

        Args:
            R: [B, 3, 3] 旋转矩阵
        Returns:
            q: [B, 4] 单位四元数 (w, x, y, z)
        """
        batch_size = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

        # Case 1: trace > 0
        s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4w
        mask = trace > 0
        q[mask, 0] = 0.25 * s[mask]
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

        # Case 2: R[0,0] is the largest diagonal
        mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s2 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
        q[mask2, 1] = 0.25 * s2[mask2]
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

        # Case 3: R[1,1] is the largest diagonal
        mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s3 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
        q[mask3, 2] = 0.25 * s3[mask3]
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

        # Case 4: R[2,2] is the largest diagonal
        mask4 = (~mask) & (~mask2) & (~mask3)
        s4 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
        q[mask4, 3] = 0.25 * s4[mask4]

        return F.normalize(q, dim=1)

    @staticmethod
    def random_rotation_matrix(batch_size, phi_max=180.0, device='cpu'):
        """
        生成随机旋转矩阵 (论文 Section VI-A 的采样方法)
        R = Exp(φ * a/||a||), 其中 a ~ N(0,I), φ ~ U[0, φ_max)

        Args:
            batch_size: 批大小
            phi_max: 最大旋转角度(度)
            device: 设备
        Returns:
            R: [B, 3, 3] 随机旋转矩阵
        """
        phi_max_rad = phi_max * 3.14159265 / 180.0

        # 随机旋转轴
        a = torch.randn(batch_size, 3, device=device)
        a = F.normalize(a, dim=1)

        # 随机旋转角度
        phi = torch.rand(batch_size, 1, device=device) * phi_max_rad

        # 轴角 → 旋转矩阵 (Rodrigues公式)
        axis_angle = a * phi  # [B, 3]
        R = axis_angle_to_rotation_matrix(axis_angle)
        return R

    @staticmethod
    def angular_error(R_pred, R_gt):
        """
        计算两个旋转矩阵之间的角度误差 (度)
        对应论文 Equation (22):
            d_ang(R, R_gt) = ||Log(R @ R_gt^T)||

        Args:
            R_pred: [B, 3, 3] 预测旋转
            R_gt: [B, 3, 3] 真值旋转
        Returns:
            error_deg: [B] 角度误差 (度)
        """
        R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        # clamp 避免数值问题
        cos_angle = torch.clamp((trace - 1.0) / 2.0, min=-1.0, max=1.0)
        angle_rad = torch.acos(cos_angle)
        return angle_rad * 180.0 / 3.14159265


def axis_angle_to_rotation_matrix(axis_angle):
    """
    轴角表示 → 旋转矩阵 (Rodrigues公式)

    Args:
        axis_angle: [B, 3] 轴角向量 (方向=旋转轴, 模=旋转角度)
    Returns:
        R: [B, 3, 3] 旋转矩阵
    """
    angle = torch.norm(axis_angle, dim=1, keepdim=True)  # [B, 1]
    axis = axis_angle / (angle + 1e-10)  # [B, 3]

    K = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0)
    angle = angle.unsqueeze(2)  # [B, 1, 1]

    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.bmm(K, K)
    return R


# ========================
# 可微分 QCQP Layer
# ========================

class QCQPLayer(nn.Module):
    """
    可微分 QCQP 层 (论文 Figure 2, Problem 3)

    输入: 网络输出的10维参数 θ
    过程: θ → A(θ) → 特征分解 → 最小特征值对应的特征向量
    输出: 单位四元数 q*

    梯度: 通过隐函数定理自动获得 (论文 Equation 9)
           PyTorch 的 torch.linalg.eigh 已内置此梯度
    """

    def __init__(self):
        super().__init__()
        self.rep = RotationRepresentations()

    def forward(self, theta):
        """
        Args:
            theta: [B, 10] 网络输出
        Returns:
            q: [B, 4] 单位四元数
            A: [B, 4, 4] 对称矩阵 (用于不确定性计算)
            eigenvalues: [B, 4] 特征值 (用于 DT 不确定性)
        """
        A = self.rep.theta_to_A(theta)
        q, eigenvalues = self.rep.A_to_quaternion(A)
        return q, A, eigenvalues
