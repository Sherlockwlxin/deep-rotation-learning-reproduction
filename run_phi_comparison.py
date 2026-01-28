"""
run_phi_comparison.py - 复现 Figure 4: 不同最大旋转角度的对比
============================================================
论文 Figure 4: Box-and-whiskers plots for φ_max = 10°, 100°, 180°
关键发现: 四元数表示在 φ_max → 180° 时误差显著增大 (不连续性导致)

运行方式:
  python run_phi_comparison.py
============================================================
"""

import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rotations import RotationRepresentations, axis_angle_to_rotation_matrix
from models import PointNetRotationNet
from losses import chordal_loss


def generate_wahba_data(batch_size, num_points, phi_max, sigma, device):
    """同 run_wahba_synthetic.py"""
    phi_max_rad = phi_max * np.pi / 180.0
    a = torch.randn(batch_size, 3, device=device)
    a = a / (torch.norm(a, dim=1, keepdim=True) + 1e-10)
    phi = torch.rand(batch_size, 1, device=device) * phi_max_rad
    axis_angle = a * phi
    R_gt = axis_angle_to_rotation_matrix(axis_angle)

    u = torch.randn(batch_size, num_points, 3, device=device)
    u = u / (torch.norm(u, dim=2, keepdim=True) + 1e-10)
    v = torch.bmm(u, R_gt.transpose(1, 2)) + sigma * torch.randn(batch_size, num_points, 3, device=device)
    data = torch.cat([u, v], dim=2)
    return data, R_gt


def train_and_evaluate(representation, lr, phi_max, epochs, device, sigma=0.01):
    """训练并返回最终测试误差"""
    model = PointNetRotationNet(representation=representation, input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rep = RotationRepresentations()

    for epoch in range(epochs):
        model.train()
        for _ in range(5):  # 5 mini-batches per epoch
            data, R_gt = generate_wahba_data(100, 100, phi_max, sigma, device)
            optimizer.zero_grad()
            output = model(data)
            loss = chordal_loss(output['R'], R_gt)
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    all_errors = []
    with torch.no_grad():
        for _ in range(20):
            data, R_gt = generate_wahba_data(100, 100, phi_max, sigma, device)
            output = model(data)
            errors = rep.angular_error(output['R'], R_gt)
            all_errors.append(errors)
    return torch.cat(all_errors).cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/phi_comparison'
    os.makedirs(output_dir, exist_ok=True)

    phi_max_values = [10, 100, 180]
    representations = ['quat', '6d', 'A']
    labels = {'quat': 'quat', '6d': '6D', 'A': 'A'}
    num_trials = 10
    epochs = 200

    # 收集结果
    results = {phi: {rep: [] for rep in representations} for phi in phi_max_values}

    for phi_max in phi_max_values:
        print(f"\n===== φ_max = {phi_max}° =====")
        for trial in range(num_trials):
            lr = 10 ** np.random.uniform(-4, -3)
            print(f"  Trial {trial+1}/{num_trials}, lr={lr:.5f}")

            for rep_name in representations:
                errors = train_and_evaluate(rep_name, lr, phi_max, epochs, device)
                results[phi_max][rep_name].append(errors)

    # ===== 绘图: 复现 Figure 4 =====
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, phi_max in enumerate(phi_max_values):
        ax = axes[i]
        box_data = []
        box_labels = []
        for rep_name in representations:
            # 将所有trial的误差合并
            all_errors = np.concatenate(results[phi_max][rep_name])
            # 取 log10 (论文的纵轴是 log10 error)
            log_errors = np.log10(np.clip(all_errors, 1e-3, None))
            box_data.append(log_errors)
            box_labels.append(labels[rep_name])

        bp = ax.boxplot(box_data, labels=box_labels, showfliers=False,
                        patch_artist=True)

        colors_box = ['#4477AA', '#66CCEE', '#EE6677']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('log₁₀ error (°)')
        ax.set_title(f'{phi_max}°')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Error vs. Maximum Rotation Angle - cf. Paper Figure 4', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phi_comparison.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {output_dir}/phi_comparison.png")

    # 打印统计
    for phi_max in phi_max_values:
        print(f"\nφ_max = {phi_max}°:")
        for rep_name in representations:
            all_err = np.concatenate(results[phi_max][rep_name])
            print(f"  {labels[rep_name]:5s}: mean={np.mean(all_err):.2f}°, "
                  f"median={np.median(all_err):.2f}°")


if __name__ == '__main__':
    main()
