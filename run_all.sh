#!/bin/bash
# ============================================================
# 一键运行所有实验
# 在AutoDL上: bash run_all.sh
# ============================================================

set -e

echo "============================================"
echo "  Deep Rotation Learning 完整复现"
echo "============================================"
echo ""

PROJECT_DIR=$(dirname "$(readlink -f "$0")")
cd $PROJECT_DIR
mkdir -p results/{wahba_synthetic,shapenet,kitti,phi_comparison}

# ============================================================
# 实验1: 合成 Wahba 问题 (论文 Figure 3)
# 预计耗时: ~1-2小时 (GPU)
# ============================================================
echo ""
echo "============================================"
echo "  实验1: Wahba Synthetic (Figure 3)"
echo "============================================"
python run_wahba_synthetic.py \
    --phi_max 180 \
    --num_trials 25 \
    --epochs 250 \
    --output_dir results/wahba_synthetic

# ============================================================
# 实验1b: φ_max 对比实验 (论文 Figure 4)
# 预计耗时: ~2-3小时 (GPU)
# ============================================================
echo ""
echo "============================================"
echo "  实验1b: Phi Comparison (Figure 4)"
echo "============================================"
python run_phi_comparison.py

# ============================================================
# 实验2: ShapeNet 点云 (论文 Figure 5)
# 预计耗时: ~2-3小时 (GPU)
# ============================================================
echo ""
echo "============================================"
echo "  实验2: ShapeNet Point Cloud (Figure 5)"
echo "============================================"
python run_shapenet.py \
    --mode synthetic \
    --num_trials 10 \
    --epochs 250 \
    --output_dir results/shapenet

# ============================================================
# 实验3: KITTI 视觉里程计 (论文 Figure 6,7, Table I)
# 预计耗时: ~1-2小时 (GPU)
# ============================================================
echo ""
echo "============================================"
echo "  实验3: KITTI Egomotion (Figure 6,7, Table I)"
echo "============================================"
python run_kitti.py \
    --mode simulate \
    --epochs 50 \
    --output_dir results/kitti

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "  所有实验完成!"
echo "============================================"
echo "结果保存在:"
echo "  results/wahba_synthetic/  - Figure 3 复现"
echo "  results/phi_comparison/   - Figure 4 复现"
echo "  results/shapenet/         - Figure 5 复现"
echo "  results/kitti/            - Figure 6,7 + Table I 复现"
echo ""
echo "可用 TensorBoard 查看日志: tensorboard --logdir logs/"
