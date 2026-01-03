
import gymnasium as gym
import numpy as np
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from src.envs.brachiation_env import BrachiationEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/brachiation_final")
    parser.add_argument("--task", type=str, default="grasping")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    # Create environment
    env = DummyVecEnv([lambda: BrachiationEnv(
        render_mode=None, 
        initial_keyframe="wall1_grip",
        task_mode=args.task
    )])

    # Load Normalization Stats
    vec_norm_path = "./checkpoints/vec_normalize.pkl"
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: No normalization stats found!")

    # Stack frames
    env = VecFrameStack(env, n_stack=4)

    # Load Model
    model = PPO.load(args.model_path, env=env)

    obs = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial observation sample (first 10):", obs[0, :10])
    
    print("\n--- Running 100 steps ---")
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Check for static action
        if i < 5:
            # Extract z-pos from obs (first 3 are x,y,z usually, but normalized?)
            # Better to check `env.envs[0].data.qpos[2]`
            z_pos = env.envs[0].data.qpos[2]
            print(f"Step {i}: Action mean={np.mean(action):.4f}, Z-pos={z_pos:.4f}")
            print(f"        Obs mean={np.mean(obs):.4f}")
            
        if done[0]:
            print(f"Episode done at step {i}")
            obs = env.reset()
            
    env.close()

if __name__ == "__main__":
    main()
