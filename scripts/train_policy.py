#!/usr/bin/env python3
"""
Training entry point for brachiation robot policies.

Supports PPO as a robust on-policy baseline plus SAC, TD3, TQC, and CrossQ for
off-policy continuous-control experiments.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import TQC, CrossQ
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    sync_envs_normalization,
)
from torch import nn

from src.envs.brachiation_env import BrachiationEnv


class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning.
    Starts from an easy position (close to goal) and gradually increases difficulty.
    """

    def __init__(
        self,
        envs: VecNormalize,
        initial_level: int = 8,
        min_level: int = 0,
        success_threshold: float = 50.0,
        window_size: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.envs = envs
        self.current_level = initial_level
        self.min_level = min_level
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.episode_rewards: list[float] = []

    def _on_training_start(self) -> None:
        self._apply_curriculum_level()

    def _apply_curriculum_level(self) -> None:
        self.envs.env_method("set_curriculum_level", self.current_level)
        self.logger.record("curriculum/level", self.current_level)

    def _on_step(self) -> bool:
        # Track episode rewards
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

                # Check if we should increase difficulty
                if len(self.episode_rewards) >= self.window_size:
                    mean_reward = np.mean(self.episode_rewards[-self.window_size :])

                    if mean_reward > self.success_threshold and self.current_level > self.min_level:
                        self.current_level -= 1
                        self.episode_rewards = []  # Reset tracking

                        if self.verbose > 0:
                            logging.info(
                                f"Curriculum: Advancing to level {self.current_level} (harder)"
                            )

                        self._apply_curriculum_level()

        return True


class RenderCallback(BaseCallback):
    """Render a single headed environment during training."""

    def __init__(self, render_every: int = 1) -> None:
        super().__init__()
        self.render_every = max(render_every, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every == 0:
            self.training_env.env_method("render")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Args:
        initial_value: Initial learning rate

    Returns:
        Function that computes current learning rate given progress remaining (1.0 -> 0.0)
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy for the brachiation robot.")
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac", "td3", "tqc", "crossq"],
        default="ppo",
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Number of timesteps to train the policy.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Frequency (in timesteps) between evaluation runs.",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of episodes per evaluation run."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints/train_policy"),
        help="Where to store checkpoints/logs.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch the viewer briefly before training to visualize the task.",
    )
    parser.add_argument(
        "--visualize-steps",
        type=int,
        default=200,
        help="How many frames to render during the visualization stage.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable GPU acceleration by forcing CPU for Stable Baselines.",
    )
    parser.add_argument(
        "--render-training",
        action="store_true",
        help="Open the MuJoCo viewer and render the training environment.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N training callback steps when --render-training is set.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=8, help="Number of parallel environments for training."
    )
    parser.add_argument(
        "--curriculum-start",
        type=int,
        default=0,
        help="Starting curriculum level (0-9, 0=beginning).",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=10000, help="Maximum steps per episode."
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=2048, help="PPO rollout steps per environment."
    )
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Warmup steps for off-policy algorithms.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Replay buffer size for off-policy algorithms.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Training frequency in environment steps for off-policy algorithms.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient steps per update for off-policy algorithms.",
    )
    parser.add_argument(
        "--td3-action-noise", type=float, default=0.1, help="TD3 Gaussian action-noise sigma."
    )
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


def make_env(
    rank: int,
    seed: int,
    curriculum_level: int,
    max_episode_steps: int,
    render_mode: str | None = None,
) -> Callable[[], Monitor]:
    """
    Create a single environment wrapped in Monitor.

    Args:
        rank: Environment index for seed offset
        seed: Base random seed
        curriculum_level: Starting difficulty level
    """

    def _init() -> Monitor:
        env = BrachiationEnv(
            render_mode=render_mode,
            initial_keyframe="wall1_grip",
            curriculum_level=curriculum_level,
            max_episode_steps=max_episode_steps,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    set_random_seed(seed)
    return _init


def make_vec_env(
    n_envs: int,
    seed: int = 0,
    curriculum_level: int = 8,
    use_subproc: bool = True,
    max_episode_steps: int = 10000,
    render_mode: str | None = None,
) -> VecNormalize:
    """
    Create vectorized environments with observation normalization.

    Args:
        n_envs: Number of parallel environments
        seed: Random seed
        curriculum_level: Starting curriculum level
        use_subproc: Whether to use SubprocVecEnv (parallel) or DummyVecEnv (serial)
    """
    env_fns = [
        make_env(i, seed, curriculum_level, max_episode_steps, render_mode=render_mode)
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Wrap with VecNormalize for observation and reward normalization
    # This significantly improves training stability
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    return vec_env


def evaluate_model(model, env: VecNormalize, episodes: int) -> None:
    logging.info("Evaluating trained policy")
    env.training = False
    env.norm_reward = False
    returns = []
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        done = np.array([False])
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
        returns.append(total_reward)
        logging.info(f" Eval episode {ep + 1}: return={total_reward:.2f}")
    logging.info(f"Mean return over {len(returns)} episodes: {np.mean(returns):.2f}")


def create_model(args: argparse.Namespace, train_env: VecNormalize):
    device = "cpu" if args.force_cpu else "auto"
    ppo_policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256],
        },
        "activation_fn": nn.Tanh,
    }
    off_policy_kwargs = {
        "policy": "MlpPolicy",
        "env": train_env,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "gamma": 0.99,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "policy_kwargs": {"net_arch": [256, 256]},
        "verbose": 1,
        "tensorboard_log": str(args.output_dir / "tensorboard"),
        "seed": 42,
        "device": device,
    }

    if args.algo == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=ppo_policy_kwargs,
            verbose=1,
            tensorboard_log=str(args.output_dir / "tensorboard"),
            seed=42,
            device=device,
        )

    if args.algo == "sac":
        return SAC(
            tau=0.005,
            ent_coef="auto",
            **off_policy_kwargs,
        )

    if args.algo == "tqc":
        return TQC(
            tau=0.005,
            ent_coef="auto",
            **off_policy_kwargs,
        )

    if args.algo == "crossq":
        return CrossQ(
            ent_coef="auto",
            **off_policy_kwargs,
        )

    action_noise = NormalActionNoise(
        mean=np.zeros(train_env.action_space.shape[-1]),
        sigma=args.td3_action_noise * np.ones(train_env.action_space.shape[-1]),
    )
    if args.algo == "td3":
        return TD3(
            tau=0.005,
            action_noise=action_noise,
            **off_policy_kwargs,
        )

    raise ValueError(f"Unsupported algorithm: {args.algo}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.force_cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.render_training and args.n_envs != 1:
        logging.info("Forcing --n-envs 1 because headed training can render one viewer.")
        args.n_envs = 1

    if args.visualize:
        try:
            visualize_environment(args.visualize_steps)
        except Exception as exc:
            logging.warning("Visualization failed: %s", exc)

    # Create parallel training environments with normalization
    logging.info(f"Creating {args.n_envs} parallel training environments...")
    train_env = make_vec_env(
        n_envs=args.n_envs,
        seed=42,
        curriculum_level=args.curriculum_start,
        use_subproc=not args.render_training,
        max_episode_steps=args.max_episode_steps,
        render_mode="human" if args.render_training else None,
    )

    # Evaluation environment (single env, no subprocess)
    eval_env = make_vec_env(
        n_envs=1,
        seed=123,
        curriculum_level=0,  # Evaluate on hardest level
        use_subproc=False,
        max_episode_steps=args.max_episode_steps,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir / "best_model"),
        log_path=str(args.output_dir / "eval_log"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),  # Adjust for n_envs
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    curriculum_callback = CurriculumCallback(
        envs=train_env,
        initial_level=args.curriculum_start,
        min_level=0,
        success_threshold=30.0,
        window_size=50,
        verbose=1,
    )

    logging.info("Initializing %s model", args.algo.upper())
    model = create_model(args, train_env)

    logging.info(
        "Starting training for %d timesteps with %d parallel envs",
        args.total_timesteps,
        args.n_envs,
    )
    if args.algo == "ppo":
        logging.info("Effective samples per PPO update: %d", args.n_envs * args.rollout_steps)

    callbacks: list[BaseCallback] = [eval_callback, curriculum_callback]
    if args.render_training:
        callbacks.append(RenderCallback(render_every=args.render_every))

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model and normalization stats
    policy_path = args.output_dir / f"brachiation_{args.algo}_{timestamp}.zip"
    model.save(policy_path)
    train_env.save(str(args.output_dir / "vec_normalize.pkl"))
    logging.info("Policy saved to %s", policy_path)
    logging.info("Normalization stats saved to %s", args.output_dir / "vec_normalize.pkl")

    sync_envs_normalization(train_env, eval_env)
    evaluate_model(model, eval_env, episodes=args.eval_episodes)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
