#!/usr/bin/env python3
"""
Advanced PPO training for brachiation robot with:
- Optimized hyperparameters for continuous control
- Parallel environments for faster training
- Curriculum learning (start easy, progress to harder)
- Custom network architecture
- Learning rate scheduling
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.brachiation_env import BrachiationEnv


class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning.
    Starts from an easy position (close to goal) and gradually increases difficulty.
    """
    def __init__(self, 
                 envs: VecNormalize,
                 initial_level: int = 8,
                 min_level: int = 0,
                 success_threshold: float = 50.0,
                 window_size: int = 100,
                 verbose: int = 0):
        super().__init__(verbose)
        self.envs = envs
        self.current_level = initial_level
        self.min_level = min_level
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.episode_rewards: List[float] = []
        
    def _on_step(self) -> bool:
        # Track episode rewards
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                
                # Check if we should increase difficulty
                if len(self.episode_rewards) >= self.window_size:
                    mean_reward = np.mean(self.episode_rewards[-self.window_size:])
                    
                    if mean_reward > self.success_threshold and self.current_level > self.min_level:
                        self.current_level -= 1
                        self.episode_rewards = []  # Reset tracking
                        
                        if self.verbose > 0:
                            logging.info(f"Curriculum: Advancing to level {self.current_level} (harder)")
                        
                        # Update environment curriculum level
                        # Note: This requires the underlying env to support curriculum updates
                        
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
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Number of timesteps to train the policy.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Frequency (in timesteps) between evaluation runs.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes per evaluation run.")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/train_policy"), help="Where to store checkpoints/logs.")
    parser.add_argument("--visualize", action="store_true", help="Launch the viewer briefly before training to visualize the task.")
    parser.add_argument("--visualize-steps", type=int, default=200, help="How many frames to render during the visualization stage.")
    parser.add_argument("--force-cpu", action="store_true", help="Disable GPU acceleration by forcing CPU for Stable Baselines.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments for training.")
    parser.add_argument("--curriculum-start", type=int, default=8, help="Starting curriculum level (0-9, higher=easier).")
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


def make_env(rank: int, seed: int, curriculum_level: int) -> Callable[[], Monitor]:
    """
    Create a single environment wrapped in Monitor.
    
    Args:
        rank: Environment index for seed offset
        seed: Base random seed
        curriculum_level: Starting difficulty level
    """
    def _init() -> Monitor:
        env = BrachiationEnv(
            render_mode=None, 
            initial_keyframe="wall1_grip",
            curriculum_level=curriculum_level,
            max_episode_steps=1000,  # Shorter episodes for faster iteration
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    
    set_random_seed(seed)
    return _init


def make_vec_env(
    n_envs: int, 
    seed: int = 0, 
    curriculum_level: int = 8,
    use_subproc: bool = True
) -> VecNormalize:
    """
    Create vectorized environments with observation normalization.
    
    Args:
        n_envs: Number of parallel environments
        seed: Random seed
        curriculum_level: Starting curriculum level
        use_subproc: Whether to use SubprocVecEnv (parallel) or DummyVecEnv (serial)
    """
    env_fns = [make_env(i, seed, curriculum_level) for i in range(n_envs)]
    
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

    # Create parallel training environments with normalization
    logging.info(f"Creating {args.n_envs} parallel training environments...")
    train_env = make_vec_env(
        n_envs=args.n_envs,
        seed=42,
        curriculum_level=args.curriculum_start,
        use_subproc=True,
    )
    
    # Evaluation environment (single env, no subprocess)
    eval_env = make_vec_env(
        n_envs=1,
        seed=123,
        curriculum_level=0,  # Evaluate on hardest level
        use_subproc=False,
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

    # PPO with optimized hyperparameters for continuous control
    logging.info("Initializing PPO with optimized hyperparameters...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        # Optimized hyperparameters
        learning_rate=linear_schedule(3e-4),  # Scheduled learning rate
        n_steps=2048,  # Steps per update (larger = more stable)
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Epochs per update
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda for advantage estimation
        clip_range=0.2,  # PPO clip range
        clip_range_vf=None,  # Don't clip value function
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        # Network architecture
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],  # Policy network
                "vf": [256, 256],  # Value network
            },
            "activation_fn": __import__("torch").nn.Tanh,  # Tanh works well for continuous control
        },
        verbose=1,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        seed=42,
    )

    logging.info("Starting training for %d timesteps with %d parallel envs", args.total_timesteps, args.n_envs)
    logging.info("Effective samples per update: %d", args.n_envs * 2048)
    
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=[eval_callback, curriculum_callback],
        progress_bar=True,
    )

    # Save final model and normalization stats
    policy_path = args.output_dir / f"brachiation_policy_{timestamp}.zip"
    model.save(policy_path)
    train_env.save(str(args.output_dir / "vec_normalize.pkl"))
    logging.info("Policy saved to %s", policy_path)
    logging.info("Normalization stats saved to %s", args.output_dir / "vec_normalize.pkl")

    evaluate_model(model, episodes=args.eval_episodes)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
