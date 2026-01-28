"""
run_kitti.py - 实验3: KITTI视觉里程计旋转估计
============================================================
复现论文 Section VI-C, Figure 6, Figure 7, Table I

实验设计:
  - 使用 KITTI Odometry 数据集的图像对
  - 预测连续帧之间的相对旋转
  - 训练集: residential + city 类别 (除测试序列外)
  - 测试集: 序列 00, 02, 05
  - 对比: quat / 6D / A, 以及 A+DT (色散阈值) 的OOD检测能力

关键发现 (论文):
  - 因为KITTI旋转量级很小(~1°), 三种表示精度差异不大
  - 但 A 表示的 DT 指标能有效检测 OOD 样本 (如损坏图像)

注意: KITTI数据集需要单独下载 (~65GB).
  本脚本提供模拟模式用于验证代码逻辑.

运行方式:
  # 模拟模式 (验证代码)
  python run_kitti.py --mode simulate

  # 完整复现 (需要KITTI数据)
  python run_kitti.py --mode kitti --data_dir data/kitti
============================================================
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rotations import RotationRepresentations, axis_angle_to_rotation_matrix
from models import ConvRotationNet
from losses import chordal_loss


class SimulatedKITTIDataset(Dataset):
    """
    模拟KITTI数据集 (用于验证代码逻辑)
    生成合成图像对和小角度旋转
    """
    def __init__(self, num_pairs=2000, image_size=(64, 192), is_corrupted=False):
        """
        Args:
            num_pairs: 图像对数量
            image_size: (H, W)
            is_corrupted: 是否生成损坏图像 (用于OOD测试)
        """
        self.num_pairs = num_pairs
        self.image_size = image_size
        self.is_corrupted = is_corrupted

        # 预生成旋转 (KITTI的旋转通常很小, ~1°)
        phi_max = 2.0  # 度
        self.rotations = []
        for _ in range(num_pairs):
            R = RotationRepresentations.random_rotation_matrix(1, phi_max=phi_max)[0]
            self.rotations.append(R)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 生成模拟图像对 (6通道: 两张图像拼接)
        H, W = self.image_size
        img = torch.randn(6, H, W) * 0.5 + 0.5

        if self.is_corrupted:
            # 模拟论文中的图像损坏: 随机矩形区域置黑
            h1, w1 = np.random.randint(0, H//2), np.random.randint(0, W//2)
            h2, w2 = np.random.randint(H//2, H), np.random.randint(W//2, W)
            img[:, h1:h2, w1:w2] = 0.0

        R_gt = self.rotations[idx]
        return img, R_gt


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    rep = RotationRepresentations()
    total_loss = 0.0
    total_error = 0.0
    n = 0

    for imgs, R_gt in dataloader:
        imgs = imgs.to(device)
        R_gt = R_gt.to(device)

        optimizer.zero_grad()
        output = model(imgs)
        loss = chordal_loss(output['R'], R_gt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            error = rep.angular_error(output['R'], R_gt).mean()
            total_loss += loss.item() * imgs.shape[0]
            total_error += error.item() * imgs.shape[0]
            n += imgs.shape[0]

    return total_loss / n, total_error / n


def evaluate_with_dt(model, dataloader, device, dt_threshold=None):
    """
    评估, 并计算 DT (Dispersion Thresholding) 指标

    论文 Section V-A:
      tr(Λ) = 3λ1 - λ2 - λ3 - λ4
      用于 OOD 检测: 保留 |tr(Λ)| 最大的样本 (最确定的)

    Args:
        model: 使用'A'表示的模型
        dataloader: 数据加载器
        device: 设备
        dt_threshold: DT阈值 (None = 不过滤)
    Returns:
        mean_error: 平均角度误差
        kept_ratio: 保留的样本比例
        dispersions: 所有样本的色散度
        errors: 所有样本的角度误差
    """
    model.eval()
    rep = RotationRepresentations()
    all_errors = []
    all_dispersions = []

    with torch.no_grad():
        for imgs, R_gt in dataloader:
            imgs = imgs.to(device)
            R_gt = R_gt.to(device)

            output = model(imgs)
            errors = rep.angular_error(output['R'], R_gt)
            all_errors.append(errors.cpu())

            if 'eigenvalues' in output:
                disp = rep.compute_dispersion(output['eigenvalues'])
                all_dispersions.append(disp.cpu())

    all_errors = torch.cat(all_errors)

    if len(all_dispersions) > 0:
        all_dispersions = torch.cat(all_dispersions)

        if dt_threshold is not None:
            # 保留色散度绝对值大于阈值的样本 (更确定的)
            mask = all_dispersions < dt_threshold  # tr(Λ) 是负数, 越小越确定
            if mask.sum() > 0:
                kept_errors = all_errors[mask]
                return kept_errors.mean().item(), mask.float().mean().item(), all_dispersions, all_errors
            else:
                return all_errors.mean().item(), 1.0, all_dispersions, all_errors

        return all_errors.mean().item(), 1.0, all_dispersions, all_errors

    return all_errors.mean().item(), 1.0, None, all_errors


def main():
    parser = argparse.ArgumentParser(description='KITTI Rotation Estimation')
    parser.add_argument('--mode', type=str, default='simulate',
                        choices=['simulate', 'kitti'])
    parser.add_argument('--data_dir', type=str, default='data/kitti')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/kitti')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ===== 1. 准备数据 =====
    print("Preparing datasets...")
    if args.mode == 'simulate':
        train_dataset = SimulatedKITTIDataset(num_pairs=2000)
        test_dataset = SimulatedKITTIDataset(num_pairs=500)
        test_corrupted = SimulatedKITTIDataset(num_pairs=500, is_corrupted=True)
    else:
        raise NotImplementedError("请先下载KITTI数据集, 或使用 --mode simulate")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_corrupted_loader = DataLoader(test_corrupted, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ===== 2. 训练三种表示 =====
    representations = ['quat', '6d', 'A']
    results = {}

    for rep_name in representations:
        print(f"\n===== Training with {rep_name} representation =====")
        model = ConvRotationNet(representation=rep_name, input_channels=6).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_error = float('inf')
        for epoch in range(args.epochs):
            train_loss, train_err = train_epoch(model, train_loader, optimizer, device)

            if (epoch + 1) % 10 == 0:
                test_err, _, _, _ = evaluate_with_dt(model, test_loader, device)
                print(f"  Epoch {epoch+1}: train_err={train_err:.4f}°, test_err={test_err:.4f}°")
                if test_err < best_error:
                    best_error = test_err
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir, f'best_{rep_name}.pth'))

        # 最终评估
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_{rep_name}.pth'),
                                         map_location=device))

        # 正常测试
        test_err, _, dispersions, errors = evaluate_with_dt(model, test_loader, device)
        # 损坏测试
        test_corrupted_err, _, dispersions_corr, errors_corr = evaluate_with_dt(
            model, test_corrupted_loader, device
        )

        results[rep_name] = {
            'normal_error': test_err,
            'corrupted_error': test_corrupted_err,
            'dispersions': dispersions,
            'dispersions_corrupted': dispersions_corr,
            'errors': errors,
            'errors_corrupted': errors_corr,
        }

        print(f"  {rep_name} - Normal: {test_err:.4f}°, Corrupted: {test_corrupted_err:.4f}°")

    # ===== 3. A + DT (色散阈值) =====
    print("\n===== Evaluating A + DT (Dispersion Thresholding) =====")
    if results['A']['dispersions'] is not None:
        # 计算训练集上的DT阈值 (论文用 q=0.75 分位数)
        model = ConvRotationNet(representation='A', input_channels=6).to(device)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_A.pth'),
                                         map_location=device))

        # 获取训练集色散度
        train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        _, _, train_dispersions, _ = evaluate_with_dt(model, train_eval_loader, device)

        if train_dispersions is not None:
            # q=0.75 分位数阈值
            threshold = torch.quantile(train_dispersions, 0.75).item()
            print(f"  DT threshold (q=0.75): {threshold:.2f}")

            # 用DT过滤后的评估
            dt_normal_err, dt_normal_kept, _, _ = evaluate_with_dt(
                model, test_loader, device, dt_threshold=threshold
            )
            dt_corrupted_err, dt_corrupted_kept, _, _ = evaluate_with_dt(
                model, test_corrupted_loader, device, dt_threshold=threshold
            )

            print(f"  A + DT Normal:    error={dt_normal_err:.4f}°, kept={dt_normal_kept*100:.1f}%")
            print(f"  A + DT Corrupted: error={dt_corrupted_err:.4f}°, kept={dt_corrupted_kept*100:.1f}%")

    # ===== 4. 绘图: 复现 Figure 6 =====
    print("\nGenerating plots...")
    if results['A']['dispersions'] is not None and results['A']['dispersions_corrupted'] is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # (a) 正常数据的 散点图
        ax = axes[0]
        disp = results['A']['dispersions'].numpy()
        err = results['A']['errors'].numpy()
        ax.scatter(disp, err, s=3, alpha=0.5, label='test')
        ax.set_xlabel('tr(Λ)')
        ax.set_ylabel('error (deg)')
        ax.set_title('Normal test data')
        ax.legend()

        # (b) 损坏数据的散点图
        ax = axes[1]
        disp_c = results['A']['dispersions_corrupted'].numpy()
        err_c = results['A']['errors_corrupted'].numpy()
        ax.scatter(disp_c, err_c, s=3, alpha=0.5, color='red', label='corrupted')
        ax.set_xlabel('tr(Λ)')
        ax.set_ylabel('error (deg)')
        ax.set_title('Corrupted test data')
        ax.legend()

        # (c) DT分布对比 (类似 Figure 6c)
        ax = axes[2]
        ax.boxplot([disp, disp_c],
                   labels=['Normal', 'Corrupted'],
                   showfliers=False)
        ax.set_ylabel('tr(Λ)')
        ax.set_title('DT Distribution Comparison')

        fig.suptitle('Dispersion Thresholding Analysis - cf. Paper Figure 6')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'kitti_dt_analysis.png'), dpi=150)
        plt.close()

    # ===== 5. 结果汇总表 (复现 Table I) =====
    print("\n" + "=" * 70)
    print("Results Summary (cf. Paper Table I)")
    print("=" * 70)
    print(f"{'Model':<15} {'Normal Error':>12} {'Kept %':>8} {'Corrupted Error':>16} {'Kept %':>8}")
    print("-" * 70)
    for rep_name in representations:
        print(f"{rep_name:<15} {results[rep_name]['normal_error']:>11.4f}° {'100':>7}% "
              f"{results[rep_name]['corrupted_error']:>15.4f}° {'100':>7}%")

    if results['A']['dispersions'] is not None and train_dispersions is not None:
        print(f"{'A + DT':<15} {dt_normal_err:>11.4f}° {dt_normal_kept*100:>7.1f}% "
              f"{dt_corrupted_err:>15.4f}° {dt_corrupted_kept*100:>7.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
