#!/bin/bash
# Train and test FCN-ResNet50 at 20%, 50%, 100% labeled data ratios

CKPT_BASE="/home/fym/Nas/fym/datasets/graduation/resnet"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for RATIO in 20 50 100; do
    CKPT_DIR="${CKPT_BASE}/supervised_${RATIO}"

    echo "============================================"
    echo "Training FCN-ResNet50 with ${RATIO}% labeled data"
    echo "Checkpoint dir: ${CKPT_DIR}"
    echo "============================================"

    python "${SCRIPT_DIR}/train.py" \
        --ratio "${RATIO}" \
        --ckpt_dir "${CKPT_DIR}" \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --val_interval 5 \
        --num_workers 4

    echo ""
    echo "Testing ${RATIO}% model..."
    python "${SCRIPT_DIR}/test.py" \
        --ratio "${RATIO}" \
        --checkpoint "${CKPT_DIR}/best.pth" \
        --batch_size 8

    echo ""
done
