#!/usr/bin/env python3
"""Train a policy that drives the brachiation robot to its goal while previewing the task."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.brachiation_env import BrachiationEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy for the brachiation robot.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Number of timesteps to train the policy.")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Frequency (in timesteps) between evaluation runs.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes per evaluation run.")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/train_policy"), help="Where to store checkpoints/logs.")
    parser.add_argument("--visualize", action="store_true", help="Launch the viewer briefly before training to visualize the task.")
    parser.add_argument("--visualize-steps", type=int, default=200, help="How many frames to render during the visualization stage.")
    parser.add_argument("--force-cpu", action="store_true", help="Disable GPU acceleration by forcing CPU for Stable Baselines.")
    return parser.parse_args()


def visualize_environment(steps: int) -> None:
    logging.info("Visualizing the brachiation task before training.")
    env = BrachiationEnv(render_mode="human", initial_keyframe="wall1_grip")
    obs, _ = env.reset()

    for frame in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


def make_vec_env(render_mode: Optional[str], monitor_path: Path) -> DummyVecEnv:
    monitor_path.parent.mkdir(parents=True, exist_ok=True)

    def make_single_env() -> Monitor:
        env = BrachiationEnv(render_mode=render_mode, initial_keyframe="wall1_grip")
        return Monitor(env, filename=str(monitor_path))

    return DummyVecEnv([make_single_env])


def evaluate_model(model: PPO, episodes: int) -> None:
    logging.info("Evaluating trained policy")
    returns = []
    env = BrachiationEnv(render_mode=None, initial_keyframe="wall1_grip")
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        returns.append(total_reward)
        logging.info(f" Eval episode {ep + 1}: return={total_reward:.2f}")
    env.close()
    logging.info(f"Mean return over {len(returns)} episodes: {np.mean(returns):.2f}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.force_cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.visualize:
        try:
            visualize_environment(args.visualize_steps)
        except Exception as exc:
            logging.warning("Visualization failed: %s", exc)

    train_monitor = args.output_dir / "monitor_train.csv"
    eval_monitor = args.output_dir / "monitor_eval.csv"

    train_env = make_vec_env(render_mode=None, monitor_path=train_monitor)
    eval_env = make_vec_env(render_mode=None, monitor_path=eval_monitor)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir / "best_model"),
        log_path=str(args.output_dir / "eval_log"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=str(args.output_dir / "tensorboard"),
    )

    logging.info("Starting training for %d timesteps", args.total_timesteps)
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    policy_path = args.output_dir / f"brachiation_policy_{timestamp}.zip"
    model.save(policy_path)
    logging.info("Policy saved to %s", policy_path)

    evaluate_model(model, episodes=args.eval_episodes)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
