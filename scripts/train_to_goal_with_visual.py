#!/usr/bin/env python3
"""Train and visualize an RL policy that reaches the goal beyond the wall course."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.envs.brachiation_env import BrachiationEnv

CHECKPOINT_DIR = ROOT / "checkpoints" / "goal_trainer"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(render_mode: str | None = None, seed: int = 0):
    def _init() -> BrachiationEnv:
        env = BrachiationEnv(render_mode=render_mode, initial_keyframe="wall1_grip")
        env.reset(seed=seed)
        return env
    return _init


def visualize_policy(
    policy: PPO | None = None,
    steps: int = 400,
    seed: int = 0,
):
    """Render several steps of either random actions or the learned policy."""
    env = BrachiationEnv(render_mode="human", initial_keyframe="wall1_grip")
    obs, _ = env.reset(seed=seed)

    for _ in range(steps):
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            obs, _ = env.reset(seed=seed)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL policy to reach the target and visualize training steps.")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--visual-steps", type=int, default=200, help="Frames to render when visualizing")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("\nVisualizing the starting configuration with random actions...\n")
    visualize_policy(policy=None, steps=args.visual_steps, seed=args.seed)

    train_env = DummyVecEnv([make_env(render_mode=None, seed=args.seed)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="auto",
        tensorboard_log=str(ROOT / "logs" / "goal_trainer"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="goal",
        verbose=1,
    )

    print(f"\nTraining for {args.timesteps} timesteps (visualization paused during learning)...\n")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    model_path = CHECKPOINT_DIR / "goal_model.zip"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    print("\nVisualizing the learned policy for a few episodes...\n")
    visualize_policy(policy=model, steps=args.visual_steps, seed=args.seed)

    train_env.close()


if __name__ == "__main__":
    main()
