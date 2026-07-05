# Dual-Arm Brachiation Robot

A reinforcement learning project for training a dual-arm robot to traverse walls using brachiation (swing) locomotion with MuJoCo physics simulation.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Smoke Test The Environment

```bash
python scripts/smoke_check.py --steps 10
pytest -q
```

### 3. Train the Robot

```bash
# Curriculum Training (Recommended)
# Start with Level 8 (easy: 2 walls to goal), decrease level as agent improves

# Stage 1: Easy (Wall 8 → Goal)
python scripts/train_policy.py --algo ppo --curriculum-start 8 --total-timesteps 500000 --n-envs 8

# Stage 2: Medium (Wall 4 → Goal)
python scripts/train_policy.py --algo ppo --curriculum-start 4 --total-timesteps 1000000 --n-envs 8

# Stage 3: Hard (Wall 0 → Goal - full course)
python scripts/train_policy.py --algo ppo --curriculum-start 0 --total-timesteps 5000000 --n-envs 8

# Off-policy continuous-control experiments
python scripts/train_policy.py --algo sac --curriculum-start 8 --total-timesteps 500000 --n-envs 4
python scripts/train_policy.py --algo td3 --curriculum-start 8 --total-timesteps 500000 --n-envs 4
python scripts/train_policy.py --algo tqc --curriculum-start 8 --total-timesteps 500000 --n-envs 4
python scripts/train_policy.py --algo crossq --curriculum-start 8 --total-timesteps 500000 --n-envs 4
```

### 4. Evaluate

```bash
# Watch trained agent
python src/main.py --mode eval --model-path ./checkpoints/brachiation_final

# Record video
python src/main.py --mode eval --model-path ./checkpoints/brachiation_final --record-video
```

### 5. Monitor Training

```bash
tensorboard --logdir ./logs/tensorboard
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | `ppo` | Training algorithm: `ppo`, `sac`, `td3`, `tqc`, or `crossq` |
| `--curriculum-start` | `0` | Starting wall (0-9). Higher = closer to goal = easier |
| `--total-timesteps` | `500000` | Training duration |
| `--n-envs` | `8` | Parallel environments |
| `--max-episode-steps` | `10000` | Episode time limit |
| `--rollout-steps` | `2048` | PPO rollout length |
| `--learning-starts` | `1000` | Off-policy replay warmup steps |
| `--output-dir` | `./checkpoints/train_policy` | Checkpoints, logs, and normalization stats |

## Project Structure

```
dual2/
├── src/
│   ├── main.py                 # Entry point
│   └── envs/
│       └── brachiation_env.py  # Gym environment
├── mujoco/
│   └── robot.xml               # MuJoCo model
├── tests/                      # Contract tests and smoke coverage
├── checkpoints/                # Saved models
├── logs/                       # Tensorboard logs
├── pyproject.toml              # Package metadata and dev dependencies
└── requirements.txt            # Legacy pip requirements
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

- Python 3.10+
- MuJoCo
- Stable-Baselines3
- Gymnasium
