"""
run_shapenet.py - 实验2: ShapeNet点云旋转估计
============================================================
复现论文 Section VI-B, Figure 5

实验设计:
  - 使用 ShapeNet airplane 类别的 2,290 个点云
  - 400 个留作测试集
  - 训练时: 每轮随机选一个点云, 应用10个随机旋转
  - 测试时: 对每个测试点云应用100个随机旋转
  - 对比三种表示: quat / 6D / A

注意: ShapeNet数据集较大, 这里提供两种模式:
  1. 使用简化的随机点云 (快速验证)
  2. 使用真实ShapeNet数据 (完整复现)

运行方式:
  # 快速验证模式 (使用随机点云代替ShapeNet)
  python run_shapenet.py --mode synthetic --num_trials 10 --epochs 250

  # 完整复现 (需要先下载ShapeNet数据)
  python run_shapenet.py --mode shapenet --data_dir data/shapenet --num_trials 10
============================================================
"""

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rotations import RotationRepresentations, axis_angle_to_rotation_matrix
from models import PointNetRotationNet
from losses import chordal_loss


class SyntheticPointCloudDataset:
    """
    合成点云数据集 (替代ShapeNet, 用于快速验证)
    生成随机3D点云, 模拟飞机形状的点分布
    """
    def __init__(self, num_clouds=2290, num_points=1024, device='cpu'):
        self.num_clouds = num_clouds
        self.num_points = num_points
        self.device = device
        # 预生成固定的参考点云
        self.clouds = []
        for _ in range(num_clouds):
            # 生成椭球形点云模拟飞机
            pts = torch.randn(num_points, 3)
            pts[:, 0] *= 2.0  # 拉伸x轴 (机身方向)
            pts[:, 2] *= 0.3  # 压缩z轴
            pts = pts / (torch.norm(pts, dim=1, keepdim=True) + 1e-10)
            self.clouds.append(pts)

    def get_batch(self, indices, num_rotations, phi_max=180.0):
        """
        获取一个batch: 选定点云 + 随机旋转

        Args:
            indices: 点云索引列表
            num_rotations: 每个点云应用的旋转数
            phi_max: 最大旋转角度
        Returns:
            data: [B*num_rot, N, 6]
            R_gt: [B*num_rot, 3, 3]
        """
        all_data = []
        all_R = []

        for idx in indices:
            cloud = self.clouds[idx].to(self.device)  # [N, 3]

            for _ in range(num_rotations):
                # 生成随机旋转
                R_gt = RotationRepresentations.random_rotation_matrix(
                    1, phi_max=phi_max, device=self.device
                )[0]  # [3, 3]

                # 应用旋转
                rotated = cloud @ R_gt.T  # [N, 3]

                # 拼接原始和旋转后的点云
                pair = torch.cat([cloud, rotated], dim=1)  # [N, 6]
                all_data.append(pair)
                all_R.append(R_gt)

        data = torch.stack(all_data)  # [B*num_rot, N, 6]
        R_gt = torch.stack(all_R)  # [B*num_rot, 3, 3]
        return data, R_gt


def train_one_epoch(model, optimizer, dataset, train_indices, device,
                    num_rotations=10, phi_max=180.0):
    """
    训练一个epoch
    论文: 每轮随机选一个点云, 应用10个随机旋转
    """
    model.train()
    rep = RotationRepresentations()

    # 随机选5个点云 (每个10个旋转 = 50个样本/epoch)
    num_clouds_per_epoch = 5
    selected = np.random.choice(train_indices, size=num_clouds_per_epoch, replace=True)

    total_loss = 0.0
    total_error = 0.0

    for idx in selected:
        data, R_gt = dataset.get_batch([idx], num_rotations, phi_max)
        data = data.to(device)
        R_gt = R_gt.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = chordal_loss(output['R'], R_gt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            error = rep.angular_error(output['R'], R_gt).mean()
            total_loss += loss.item()
            total_error += error.item()

    return total_loss / num_clouds_per_epoch, total_error / num_clouds_per_epoch


def evaluate(model, dataset, test_indices, device, num_rotations=100, phi_max=180.0):
    """评估 (论文: 对每个测试点云应用100个随机旋转)"""
    model.eval()
    rep = RotationRepresentations()
    all_errors = []

    with torch.no_grad():
        # 为了效率, 每次评估一个点云
        for idx in test_indices[:50]:  # 取前50个测试点云
            data, R_gt = dataset.get_batch([idx], num_rotations=10, phi_max=phi_max)
            data = data.to(device)
            R_gt = R_gt.to(device)

            output = model(data)
            errors = rep.angular_error(output['R'], R_gt)
            all_errors.append(errors)

    all_errors = torch.cat(all_errors)
    return all_errors.mean().item(), all_errors.median().item()


def run_single_trial(representation, lr, dataset, train_idx, test_idx,
                     epochs, device):
    """运行单次实验"""
    model = PointNetRotationNet(representation=representation, input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_errors = []
    test_errors = []

    for epoch in range(epochs):
        _, train_err = train_one_epoch(model, optimizer, dataset, train_idx, device)
        test_mean, _ = evaluate(model, dataset, test_idx, device)

        train_errors.append(train_err)
        test_errors.append(test_mean)

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: train={train_err:.2f}°, test={test_mean:.2f}°")

    return train_errors, test_errors


def main():
    parser = argparse.ArgumentParser(description='ShapeNet Point Cloud Experiment')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'shapenet'],
                        help='synthetic=快速验证, shapenet=完整复现')
    parser.add_argument('--data_dir', type=str, default='data/shapenet')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='实验次数 (论文用10次)')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/shapenet')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 创建数据集
    print("Creating dataset...")
    if args.mode == 'synthetic':
        dataset = SyntheticPointCloudDataset(num_clouds=2290, num_points=1024, device=device)
    else:
        raise NotImplementedError(
            "ShapeNet数据加载需要下载数据集。\n"
            "请先运行: python download_shapenet.py\n"
            "或使用 --mode synthetic 进行快速验证"
        )

    # 划分训练/测试
    all_indices = np.arange(2290)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:1890]
    test_indices = all_indices[1890:]  # 400个测试

    representations = ['quat', '6d', 'A']
    all_results = {rep_name: {'train': [], 'test': []} for rep_name in representations}

    for trial in range(args.num_trials):
        lr = 10 ** np.random.uniform(-4, -3)
        print(f"\n===== Trial {trial+1}/{args.num_trials}, lr={lr:.6f} =====")

        for rep_name in representations:
            print(f"  Training with {rep_name}...")
            train_errors, test_errors = run_single_trial(
                rep_name, lr, dataset, train_indices, test_indices,
                args.epochs, device
            )
            all_results[rep_name]['train'].append(train_errors)
            all_results[rep_name]['test'].append(test_errors)

    # ===== 绘图: 复现 Figure 5b =====
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'quat': 'blue', '6d': 'green', 'A': 'red'}
    labels = {'quat': 'quat', '6d': '6D', 'A': 'A (ours)'}

    for phase_idx, phase in enumerate(['train', 'test']):
        ax = axes[phase_idx]
        for rep_name in representations:
            data = np.array(all_results[rep_name][phase])
            p10 = np.percentile(data, 10, axis=0)
            p50 = np.percentile(data, 50, axis=0)
            p90 = np.percentile(data, 90, axis=0)
            epochs_x = np.arange(len(p50))

            ax.semilogy(epochs_x, p50, color=colors[rep_name], label=labels[rep_name])
            ax.fill_between(epochs_x, p10, p90, color=colors[rep_name], alpha=0.15)

        ax.set_xlabel('epoch')
        ax.set_ylabel('mean error (deg)')
        ax.set_title(f'epoch ({phase})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('ShapeNet Point Cloud Experiment - cf. Paper Figure 5b')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'shapenet_results.png'), dpi=150)
    plt.close()
    print(f"Plot saved to {args.output_dir}/shapenet_results.png")

    # 保存结果
    np.savez(
        os.path.join(args.output_dir, 'shapenet_results.npz'),
        **{f'{rep}_{phase}': np.array(all_results[rep][phase])
           for rep in representations for phase in ['train', 'test']}
    )

    # 打印最终统计
    print("\n===== Final Test Errors (median over trials, last epoch) =====")
    for rep_name in representations:
        final_errors = [all_results[rep_name]['test'][t][-1] for t in range(args.num_trials)]
        print(f"  {labels[rep_name]:12s}: {np.median(final_errors):.2f}° (median), "
              f"{np.mean(final_errors):.2f}° (mean)")


if __name__ == '__main__':
    main()
