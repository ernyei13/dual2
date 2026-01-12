#!/usr/bin/env python3
"""
Train a policy for grasping and balancing on a bar.

This simpler task teaches the robot to:
1. Grip the bar firmly
2. Hold itself up
3. Stay stable without falling
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.brachiation_env import BrachiationEnv


def make_env(rank: int, seed: int = 0):
    """Create a single environment instance."""
    def _init():
        env = BrachiationEnv(
            task_mode="grasping",  # Use grasping reward
            max_episode_steps=1000,  # Shorter episodes for grasping
            curriculum_level=0,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class GraspingCallback(BaseCallback):
    """Callback to log grasping-specific metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_grips = []
        
    def _on_step(self) -> bool:
        # Log grip success rate
        for info in self.locals.get("infos", []):
            if "grip_reward" in info:
                self.episode_grips.append(info["grip_reward"] > 0)
                
        if self.n_calls % 5000 == 0 and len(self.episode_grips) > 0:
            grip_rate = np.mean(self.episode_grips[-1000:])
            print(f"  Grip success rate (last 1000 steps): {grip_rate:.2%}")
            
        return True


def main():
    parser = argparse.ArgumentParser(description="Train grasping policy")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/grasping")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GRASPING POLICY TRAINING")
    print("=" * 60)
    print(f"Training for {args.total_timesteps} timesteps")
    print(f"Using {args.n_envs} parallel environments")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Create environments
    print("\nCreating environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(100)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # PPO with parameters tuned for stability
    device = "cpu" if args.force_cpu else "auto"
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Higher entropy for exploration
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        device=device,
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_log"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    grasp_callback = GraspingCallback()
    
    # Train
    print("\nStarting training...")
    print("Task: Learn to GRIP and BALANCE on the bar")
    print("Reward: +3 per grip contact, +1 height, +0.5 stability, -5 fall")
    print()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, grasp_callback],
        progress_bar=True,
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"grasping_policy_{timestamp}.zip"
    model.save(str(model_path))
    print(f"\nPolicy saved to {model_path}")
    
    # Save normalization stats
    env.save(str(output_dir / "vec_normalize.pkl"))
    print(f"Normalization stats saved")
    
    # Final evaluation
    print("\nFinal evaluation:")
    obs = eval_env.reset()
    total_reward = 0
    total_grip = 0
    steps = 0
    
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward[0]
        if info[0].get("grip_reward", 0) > 0:
            total_grip += 1
        steps += 1
        if done[0]:
            break
    
    print(f"  Survived {steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Grip rate: {total_grip/steps:.2%}")
    
    env.close()
    eval_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
