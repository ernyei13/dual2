# Research And Training Strategy

Date: 2026-07-05

## Current Project State

`dual2` is now a runnable MuJoCo/Gymnasium research repo with:

- an installable Python package,
- Gymnasium/SB3-compatible environment contract tests,
- a headless smoke check,
- CI,
- PPO, SAC, TD3, TQC, and CrossQ training options,
- curriculum level propagation to vectorized training environments,
- normalized policy actions mapped to MuJoCo actuator control ranges.

The robot is not yet known to solve full-course brachiation. The current target is to build reliable evidence: train staged tasks, evaluate them deterministically, and compare algorithms with the same environment, wrappers, seeds, and metrics.

## Local Machine Assessment

The available local machine is an Apple M1 Pro MacBook Pro with 8 CPU cores, 16 GB memory, and a 14-core Apple GPU. PyTorch MPS is available, but Stable-Baselines3's own PPO guidance recommends CPU plus vectorized environments for non-CNN policies. This machine is suitable for:

- smoke tests,
- short training runs,
- algorithm comparisons at small budgets,
- reward/curriculum debugging,
- overnight CPU runs with modest `n-envs`.

It is not the right target for massive SOTA-scale sweeps, thousands of parallel GPU environments, or MJX/Isaac Lab throughput experiments. For those, use a Linux workstation or cloud machine with an NVIDIA GPU.

## Algorithm Tracks

### Track A: PPO Baseline

PPO remains the first baseline because it is stable, simple, vectorized, and commonly used for locomotion-style control. It is also forgiving while the environment and reward are still changing.

Use it for:

- first reproducible stage baselines,
- curriculum debugging,
- reward shaping comparisons,
- establishing deterministic evaluation metrics.

### Track B: SAC Off-Policy Continuous Control

SAC is the main near-term alternative. It is off-policy, entropy-regularized, and was designed to improve sample efficiency and stability on continuous-control tasks. It should be compared after each stage has a clean success metric.

Use it for:

- grip/hang stabilization,
- one-bar transfer,
- sample-efficiency comparisons against PPO,
- robustness to reward shaping changes.

### Track C: TD3 Deterministic Actor-Critic

TD3 is useful as a deterministic off-policy comparison point. It can work well for smooth continuous control, but it can be more sensitive to exploration noise and sparse rewards than SAC.

Use it for:

- deterministic policy comparison,
- ablation against SAC,
- stages where contact timing is not too sparse.

### Track D: TQC And CrossQ SB3-Contrib Baselines

TQC and CrossQ are the strongest drop-in algorithms now available in this repo. TQC adds distributional critics to the SAC/TD3 family to reduce value overestimation. CrossQ targets sample efficiency by using batch normalization in the critic and removing target networks.

Use them for:

- best near-term model-free continuous-control comparisons,
- sample-efficiency checks against SAC,
- deciding whether the current reward/curriculum is learnable before moving to heavier model-based stacks.

### Track E: SOTA-Scale Next Work

The stronger algorithmic directions are not one-line swaps:

- simplified-model imitation for brachiation,
- model-based RL such as TD-MPC2 or Dreamer-style world models,
- MJX/Brax batched simulation,
- Isaac Lab/Isaac Sim GPU-native robot learning,
- LLM-guided reward/domain-randomization workflows such as DrEureka.

These should be treated as dedicated branches once the current MuJoCo environment has stable stage metrics.

## Brachiation-Specific Lessons From Literature

Brachiation is hard because contact timing, underactuation, grasp precision, and multi-swing planning interact. The most relevant brachiation-specific direction is simplified-model guidance: first solve an easier model that produces center-of-mass and handhold timing references, then train the full model to imitate those references.

This suggests the next major environment feature should not be a bigger neural network. It should be a staged task suite:

1. grip and hang,
2. swing while maintaining one grip,
3. reach next bar,
4. transfer grip,
5. clear one interval,
6. traverse the full course.

Each stage should have success rate, mean return, episode length, falls, grip losses, contacts, and walls cleared.

## Recommended Training Commands

Fast smoke:

```bash
python scripts/train_policy.py \
  --algo ppo \
  --total-timesteps 64 \
  --n-envs 1 \
  --eval-freq 64 \
  --eval-episodes 1 \
  --force-cpu \
  --output-dir /tmp/dual2-train-smoke \
  --max-episode-steps 50 \
  --rollout-steps 32 \
  --batch-size 32
```

Local PPO baseline:

```bash
python scripts/train_policy.py \
  --algo ppo \
  --curriculum-start 8 \
  --total-timesteps 500000 \
  --n-envs 6 \
  --force-cpu \
  --output-dir ./checkpoints/ppo_level8
```

Local SAC comparison:

```bash
python scripts/train_policy.py \
  --algo sac \
  --curriculum-start 8 \
  --total-timesteps 500000 \
  --n-envs 4 \
  --force-cpu \
  --output-dir ./checkpoints/sac_level8
```

Local TD3 comparison:

```bash
python scripts/train_policy.py \
  --algo td3 \
  --curriculum-start 8 \
  --total-timesteps 500000 \
  --n-envs 4 \
  --force-cpu \
  --output-dir ./checkpoints/td3_level8
```

Local TQC comparison:

```bash
python scripts/train_policy.py \
  --algo tqc \
  --curriculum-start 8 \
  --total-timesteps 500000 \
  --n-envs 4 \
  --force-cpu \
  --output-dir ./checkpoints/tqc_level8
```

Local CrossQ comparison:

```bash
python scripts/train_policy.py \
  --algo crossq \
  --curriculum-start 8 \
  --total-timesteps 500000 \
  --n-envs 4 \
  --force-cpu \
  --output-dir ./checkpoints/crossq_level8
```

## Primary Sources

- AcroMonk: A Minimalist Underactuated Brachiating Robot: https://arxiv.org/abs/2305.08373
- Learning to Brachiate via Simplified Model Imitation: https://arxiv.org/abs/2205.03943
- Soft Actor-Critic: https://arxiv.org/abs/1801.01290
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DrEureka: https://arxiv.org/abs/2406.01967
- MuJoCo MJX documentation: https://mujoco.readthedocs.io/en/stable/mjx.html
- Stable-Baselines3 algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
- SB3-Contrib: https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html
- TQC documentation: https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html
- CrossQ documentation: https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html
- Stable-Baselines3 custom environment checker: https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html
- Isaac Lab reinforcement learning docs: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/index.html
