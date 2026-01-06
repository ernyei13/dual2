#!/bin/bash
# Training script for brachiation robot
# Run this script to train the policy and generate an evaluation video

set -e

cd "$(dirname "$0")"

echo "=== Brachiation Robot Training ==="
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import numpy, gymnasium, stable_baselines3, mujoco, imageio" 2>/dev/null || {
    echo "Installing required packages..."
    pip3 install numpy gymnasium stable-baselines3 mujoco imageio imageio-ffmpeg rich
}

echo ""
echo "Starting training..."
echo ""

# Run training with 500k timesteps
python3 scripts/train_policy.py \
    --total-timesteps 500000 \
    --eval-freq 10000 \
    --eval-episodes 5 \
    --output-dir ./checkpoints/brachiation_final

echo ""
echo "Training complete!"
echo ""

# Record evaluation video
echo "Recording evaluation video..."
python3 scripts/record_agent.py \
    --model-path ./checkpoints/brachiation_final/best_model/best_model.zip \
    --output ./eval_video.mp4 \
    --duration 15.0

echo ""
echo "=== Done! ==="
echo "Model saved to: ./checkpoints/brachiation_final/"
echo "Video saved to: ./eval_video.mp4"
