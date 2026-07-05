#!/usr/bin/env python3
"""Headless smoke check for the MuJoCo/Gymnasium environment."""

from __future__ import annotations

import argparse

import numpy as np

from src.envs.brachiation_env import BrachiationEnv


def assert_finite(name: str, value: np.ndarray | float) -> None:
    values = np.asarray(value)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains non-finite values: {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short headless environment smoke check.")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Maximum number of random steps to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic actions.",
    )
    parser.add_argument(
        "--task",
        choices=["traversal", "grasping"],
        default="traversal",
        help="Environment task mode.",
    )
    parser.add_argument(
        "--curriculum-level",
        type=int,
        default=8,
        help="Traversal curriculum start level.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = BrachiationEnv(
        render_mode=None,
        initial_keyframe="wall1_grip",
        task_mode=args.task,
        curriculum_level=args.curriculum_level,
        max_episode_steps=max(args.steps + 5, 10),
    )

    try:
        obs, info = env.reset(seed=args.seed)
        assert obs.shape == env.observation_space.shape
        assert_finite("reset observation", obs)
        assert "walls_cleared" in info

        steps_run = 0
        total_reward = 0.0
        for _ in range(args.steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_run += 1
            total_reward += float(reward)

            assert_finite("step observation", obs)
            assert_finite("reward", reward)
            assert_finite("qpos", env.data.qpos)
            assert_finite("qvel", env.data.qvel)

            if terminated or truncated:
                break

        print(
            "Smoke check passed: "
            f"steps={steps_run}, total_reward={total_reward:.3f}, "
            f"walls_cleared={info.get('walls_cleared')}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
