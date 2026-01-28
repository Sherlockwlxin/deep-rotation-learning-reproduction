"""
run_wahba_synthetic.py - 实验1: 合成Wahba问题
============================================================
复现论文 Section VI-A, Figure 3, Figure 4

实验设计:
  - 给定N对向量对 (u_i, v_i), v_i = R_hat @ u_i + epsilon
  - 网络从 (u_i, v_i) 预测旋转 R
  - 对比三种表示: quat / 6D / A (ours)

关键设置 (论文 Section VI-A):
  - u_i 从单位球面采样
  - R_hat = Exp(φ * a/||a||), a ~ N(0,I), φ ~ U[0, φ_max)
  - σ = 0.01 (噪声标准差)
  - 动态训练集: 每个mini-batch = 100个旋转, 每个100对匹配
  - 一个epoch = 5个mini-batch
  - 学习率从 {10^-4, 10^-3} 中采样 (对数均匀)
  - 使用 Adam 优化器
  - 使用 L_chord (弦距离损失)

运行方式:
  python run_wahba_synthetic.py --phi_max 180 --num_trials 25 --epochs 250
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


def generate_wahba_data(batch_size, num_points, phi_max, sigma, device):
    """
    生成 Wahba 问题的合成数据

    论文 Equation (27):
        v_i = R_hat @ u_i + epsilon_i, epsilon_i ~ N(0, σ²I)

    Args:
        batch_size: 旋转的数量
        num_points: 每个旋转对应的向量对数量
        phi_max: 最大旋转角度 (度)
        sigma: 噪声标准差
        device: 计算设备
    Returns:
        data: [B, N, 6] 拼接的 (u_i, v_i)
        R_gt: [B, 3, 3] 真值旋转矩阵
    """
    phi_max_rad = phi_max * np.pi / 180.0

    # 采样随机旋转轴 a ~ N(0, I), 归一化
    a = torch.randn(batch_size, 3, device=device)
    a = a / (torch.norm(a, dim=1, keepdim=True) + 1e-10)

    # 采样随机旋转角度 φ ~ U[0, φ_max)
    phi = torch.rand(batch_size, 1, device=device) * phi_max_rad

    # 轴角 → 旋转矩阵
    axis_angle = a * phi  # [B, 3]
    R_gt = axis_angle_to_rotation_matrix(axis_angle)  # [B, 3, 3]

    # 采样单位球面上的参考向量 u_i
    u = torch.randn(batch_size, num_points, 3, device=device)
    u = u / (torch.norm(u, dim=2, keepdim=True) + 1e-10)

    # 旋转并加噪声: v_i = R @ u_i + epsilon
    v = torch.bmm(u, R_gt.transpose(1, 2))  # [B, N, 3]
    v = v + sigma * torch.randn_like(v)

    # 拼接为网络输入 [u_i; v_i]
    data = torch.cat([u, v], dim=2)  # [B, N, 6]
    return data, R_gt


def train_one_epoch(model, optimizer, phi_max, sigma, device, num_batches=5,
                    batch_size=100, num_points=100):
    """训练一个epoch (= num_batches 个 mini-batch)"""
    model.train()
    total_loss = 0.0
    total_error = 0.0
    rep = RotationRepresentations()

    for _ in range(num_batches):
        data, R_gt = generate_wahba_data(batch_size, num_points, phi_max, sigma, device)

        optimizer.zero_grad()
        output = model(data)
        R_pred = output['R']

        loss = chordal_loss(R_pred, R_gt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            error = rep.angular_error(R_pred, R_gt).mean()
            total_loss += loss.item()
            total_error += error.item()

    return total_loss / num_batches, total_error / num_batches


def evaluate(model, phi_max, sigma, device, num_batches=10,
             batch_size=100, num_points=100):
    """在测试集上评估"""
    model.eval()
    all_errors = []
    rep = RotationRepresentations()

    with torch.no_grad():
        for _ in range(num_batches):
            data, R_gt = generate_wahba_data(batch_size, num_points, phi_max, sigma, device)
            output = model(data)
            R_pred = output['R']
            errors = rep.angular_error(R_pred, R_gt)
            all_errors.append(errors)

    all_errors = torch.cat(all_errors)
    return all_errors.mean().item(), all_errors.median().item()


def run_single_trial(representation, lr, phi_max, epochs, sigma, device):
    """运行单次实验"""
    model = PointNetRotationNet(representation=representation, input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_errors = []
    test_errors = []

    for epoch in range(epochs):
        train_loss, train_err = train_one_epoch(model, optimizer, phi_max, sigma, device)
        test_mean, test_median = evaluate(model, phi_max, sigma, device)

        train_errors.append(train_err)
        test_errors.append(test_mean)

    return train_errors, test_errors


def main():
    parser = argparse.ArgumentParser(description='Wahba Synthetic Experiment')
    parser.add_argument('--phi_max', type=float, default=180.0,
                        help='最大旋转角度(度), 论文测试 10/100/180')
    parser.add_argument('--num_trials', type=int, default=25,
                        help='实验次数 (论文用25次)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='训练epoch数')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='噪声标准差')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/wahba_synthetic')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"phi_max = {args.phi_max}°, num_trials = {args.num_trials}, epochs = {args.epochs}")

    representations = ['quat', '6d', 'A']
    all_results = {rep_name: {'train': [], 'test': []} for rep_name in representations}

    for trial in range(args.num_trials):
        # 论文: 学习率从 {10^-4, 10^-3} 对数均匀采样
        lr = 10 ** np.random.uniform(-4, -3)
        print(f"\n===== Trial {trial+1}/{args.num_trials}, lr={lr:.6f} =====")

        for rep_name in representations:
            print(f"  Training with {rep_name} representation...")
            train_errors, test_errors = run_single_trial(
                rep_name, lr, args.phi_max, args.epochs, args.sigma, device
            )
            all_results[rep_name]['train'].append(train_errors)
            all_results[rep_name]['test'].append(test_errors)

    # ===== 绘图: 复现 Figure 3 =====
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'quat': 'blue', '6d': 'green', 'A': 'red'}
    labels = {'quat': 'quat', '6d': '6D', 'A': 'A (ours)'}

    for phase_idx, phase in enumerate(['train', 'test']):
        ax = axes[phase_idx]
        for rep_name in representations:
            data = np.array(all_results[rep_name][phase])  # [num_trials, epochs]
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

    fig.suptitle(f'Wahba Synthetic (φ_max = {args.phi_max}°) - cf. Paper Figure 3')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'wahba_phi{int(args.phi_max)}.png'), dpi=150)
    plt.close()
    print(f"Plot saved to {args.output_dir}/wahba_phi{int(args.phi_max)}.png")

    # ===== 保存数值结果 =====
    np.savez(
        os.path.join(args.output_dir, f'wahba_phi{int(args.phi_max)}_results.npz'),
        **{f'{rep}_{phase}': np.array(all_results[rep][phase])
           for rep in representations for phase in ['train', 'test']}
    )
    print("Results saved.")

    # 打印最终统计
    print("\n===== Final Test Errors (median over trials, last epoch) =====")
    for rep_name in representations:
        final_errors = [all_results[rep_name]['test'][t][-1] for t in range(args.num_trials)]
        print(f"  {labels[rep_name]:12s}: {np.median(final_errors):.2f}° (median), "
              f"{np.mean(final_errors):.2f}° (mean)")


if __name__ == '__main__':
    main()
