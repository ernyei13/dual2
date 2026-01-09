#!/usr/bin/env python3
"""
Script to record a video of a trained PPO agent.
Supports VecNormalize wrapper for proper observation normalization.
"""

import argparse
import sys
import numpy as np
import imageio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.brachiation_env import BrachiationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def record_agent(model_path, output_path, duration=10.0, fps=30, vec_normalize_path=None):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print("Creating environment...")
    # Use rgb_array mode for recording
    env = BrachiationEnv(render_mode="rgb_array", curriculum_level=0)
    
    # Load VecNormalize stats if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading normalization stats from {vec_normalize_path}...")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # Don't update stats during eval
        vec_env.norm_reward = False  # Don't normalize rewards during eval
        use_vec_env = True
    else:
        use_vec_env = False
    
    if use_vec_env:
        obs = vec_env.reset()
    else:
        obs, _ = env.reset()
    
    frames = []
    n_frames = int(duration * fps)
    
    print(f"Recording {duration}s video ({n_frames} frames)...")
    
    total_reward = 0
    episode_count = 0
    
    for i in range(n_frames):
        action, _ = model.predict(obs, deterministic=True)
        
        if use_vec_env:
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            
            # Get frame from underlying env
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                
            if done[0]:
                episode_count += 1
                # VecEnv auto-resets
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                
            if terminated or truncated:
                episode_count += 1
                obs, _ = env.reset()
            
        if i % 100 == 0:
            print(f"  Frame {i}/{n_frames} | Reward: {total_reward:.2f} | Episodes: {episode_count}")

    print(f"\nTotal reward: {total_reward:.2f} over {episode_count} episodes")
    print(f"Saving video to {output_path}...")
    
    # Save at simulation fps for real-time playback
    video_fps = min(env.control_freq, 60)  # Cap at 60fps for reasonable file size
    imageio.mimsave(output_path, frames, fps=video_fps)
    print(f"Done! Video saved with {len(frames)} frames at {video_fps} FPS")
    
    if use_vec_env:
        vec_env.close()
    else:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a video of a trained agent")
    parser.add_argument("--model-path", type=str, default="./checkpoints/brachiation_final/best_model/best_model.zip",
                        help="Path to the trained model")
    parser.add_argument("--vec-normalize", type=str, default="./checkpoints/brachiation_final/vec_normalize.pkl",
                        help="Path to VecNormalize stats (optional)")
    parser.add_argument("--output", type=str, default="eval_video.mp4",
                        help="Output video path")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Video duration in seconds")
    args = parser.parse_args()
    
    record_agent(
        args.model_path, 
        args.output, 
        args.duration,
        vec_normalize_path=args.vec_normalize
    )
