# Dual-Arm Brachiation Robot

A reinforcement learning project for training a dual-arm robot to traverse walls using brachiation (swing) locomotion with MuJoCo physics simulation.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Robot

```bash
# Curriculum Training (Recommended)
# Start with Level 8 (easy: 2 walls to goal), decrease level as agent improves

# Stage 1: Easy (Wall 8 → Goal)
python src/main.py --mode train --curriculum-level 8 --total-timesteps 500000 --num-envs 16

# Stage 2: Medium (Wall 4 → Goal)
python src/main.py --mode train --curriculum-level 4 --total-timesteps 1000000 --num-envs 16

# Stage 3: Hard (Wall 0 → Goal - full course)
python src/main.py --mode train --curriculum-level 0 --total-timesteps 5000000 --num-envs 16
```

### 3. Evaluate

```bash
# Watch trained agent
python src/main.py --mode eval --model-path ./checkpoints/brachiation_final

# Record video
python src/main.py --mode eval --model-path ./checkpoints/brachiation_final --record-video
```

### 4. Monitor Training

```bash
tensorboard --logdir ./logs/tensorboard
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `viewer` | `viewer`, `demo`, `train`, `eval` |
| `--curriculum-level` | `8` | Starting wall (0-9). Higher = closer to goal = easier |
| `--total-timesteps` | `1000000` | Training duration |
| `--num-envs` | `4` | Parallel environments |
| `--headless` | `false` | No visualization (faster training) |
| `--model-path` | - | Path to model for evaluation |
| `--record-video` | `false` | Save evaluation video |

## Project Structure

```
dual2/
├── src/
│   ├── main.py                 # Entry point
│   └── envs/
│       └── brachiation_env.py  # Gym environment
├── mujoco/
│   └── robot.xml               # MuJoCo model
├── checkpoints/                # Saved models
├── logs/                       # Tensorboard logs
└── requirements.txt
```

## How It Works

**Reward**: Distance progress toward goal (`old_dist - new_dist`)  
**Curriculum**: Start close to goal, gradually increase difficulty  
**Grasp Reflex**: Gripper auto-closes on contact (focuses learning on swing)

## Tips

1. **Tensorboard** shows `distance_progress` and `walls_cleared` metrics
2. **Checkpoints** saved every 10k steps in `./checkpoints/`
3. **GPU**: Add `device="cuda"` to PPO in `main.py` if available

## Requirements

- Python 3.8+
- MuJoCo
- Stable-Baselines3
- Gymnasium
