#!/usr/bin/env python3
"""
Script to record a video of a trained PPO agent.
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

def record_agent(model_path, output_path, duration=10.0, fps=30):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print("Creating environment...")
    # Use rgb_array mode for recording
    env = BrachiationEnv(render_mode="rgb_array")
    
    obs, _ = env.reset()
    
    frames = []
    n_frames = int(duration * fps)
    # We need to manually manage the rendering timing since we are not using the env's internal loop
    # But since BrachiationEnv.render() returns the current state, we track steps.
    
    print(f"Recording {duration}s video...")
    
    total_reward = 0
    
    # Run simulation
    # Gymnasium steps logic
    for i in range(n_frames):
        # We might need multiple physics steps per frame if control freq is different from fps
        # usage: env.control_freq is 50Hz. FPS is 30Hz.
        # This simple loop captures frames at step intervals. 
        # For smooth video, we ideally step the env at 1/fps intervals.
        # BrachiationEnv step() handles frame_skip based on control_freq (50Hz).
        # So one step = 1/50 sec = 0.02s.
        # 30FPS = 0.033s.
        # We can just record every step and playback at 50fps, or subsample.
        # Let's just record every step and set video fps to 50 for 1:1 speed.
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        frame = env.render()
        if frame is not None:
             frames.append(frame)
             
        if terminated or truncated:
            obs, _ = env.reset()
            
        if i % 50 == 0:
            print(f"Step {i}/{n_frames} Reward: {total_reward:.2f}")

    print(f"Saving video to {output_path} with simulated FPS {env.control_freq}...")
    imageio.mimsave(output_path, frames, fps=env.control_freq)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/brachiation_final")
    parser.add_argument("--output", type=str, default="trained_agent_demo.mp4")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    
    record_agent(args.model_path, args.output, args.duration)
