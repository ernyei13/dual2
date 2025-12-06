#!/usr/bin/env python3
"""
Main entry point for running the dual-arm brachiation simulation with MuJoCo.

This script provides a simple way to:
1. Visualize the robot in the MuJoCo viewer
2. Run a basic simulation
3. Train with reinforcement learning

Usage:
    python main.py                    # Run interactive viewer
    python main.py --mode train       # Train with PPO
    python main.py --mode demo        # Run demo with random actions
    python main.py --headless         # Run without visualization
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_model_path() -> str:
    """Get the path to the MuJoCo model file."""
    model_path = PROJECT_ROOT / "mujoco" / "robot.xml"
    if not model_path.exists():
        raise FileNotFoundError(
            f"MuJoCo model not found at {model_path}. "
            "Make sure you're running from the project directory."
        )
    return str(model_path)


def run_viewer():
    """Run the interactive MuJoCo viewer."""
    import mujoco
    import mujoco.viewer
    
    model_path = get_model_path()
    print(f"Loading model from: {model_path}")
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print("\n=== MuJoCo Viewer Controls ===")
    print("  Mouse: Rotate/Pan/Zoom camera")
    print("  Space: Pause/Resume simulation (starts paused)")
    print("  Backspace: Reset simulation")
    print("  Tab: Toggle visualization options")
    print("  ESC: Exit viewer")
    print("=" * 32)
    
    paused = True

    def _toggle_pause(key: int) -> None:
        nonlocal paused
        if key == mujoco.viewer.glfw.KEY_SPACE:
            paused = not paused
            print("Simulation resumed." if not paused else "Simulation paused.")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=_toggle_pause,
    ) as viewer:
        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)
            viewer.sync()


def run_demo(headless: bool = False, duration: float = 10.0):
    """Run a demo with random actions."""
    import mujoco
    import numpy as np
    
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Running demo for {duration} seconds...")
    print(f"Model has {model.nu} actuators and {model.nq} generalized coordinates")
    
    if headless:
        # Headless simulation
        steps = int(duration / model.opt.timestep)
        for i in range(steps):
            # Apply random actions
            data.ctrl[:] = np.random.uniform(-0.5, 0.5, model.nu)
            mujoco.mj_step(model, data)
            
            if i % 1000 == 0:
                print(f"Step {i}/{steps}, Time: {data.time:.2f}s")
        print("Demo completed!")
    else:
        # With viewer
        import mujoco.viewer
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = data.time
            while viewer.is_running() and (data.time - start_time) < duration:
                # Apply sinusoidal motion to demonstrate joints
                t = data.time
                for i in range(model.nu):
                    data.ctrl[i] = 0.5 * np.sin(t * 2 + i * 0.5)
                
                mujoco.mj_step(model, data)
                viewer.sync()


def run_training(args):
    """Run reinforcement learning training."""
    from src.envs.brachiation_env import BrachiationEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    print("Starting PPO training...")
    
    # Create environment factory
    def make_env():
        return BrachiationEnv(
            render_mode="rgb_array" if args.headless else None,
            initial_keyframe="wall1_grip",
        )
    
    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env])
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="brachiation_ppo"
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./logs/tensorboard/",
        device="auto",
    )
    
    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # Save final model
    model.save("./checkpoints/brachiation_final")
    print("Training completed! Model saved to ./checkpoints/brachiation_final")
    
    env.close()


def run_evaluation(args):
    """Evaluate a trained model."""
    from src.envs.brachiation_env import BrachiationEnv
    from stable_baselines3 import PPO
    import numpy as np
    
    model_path = args.model_path or "./checkpoints/brachiation_final"
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = BrachiationEnv(
        render_mode="human" if not args.headless else None,
        initial_keyframe="wall1_grip",
    )
    
    print("Running evaluation...")
    total_rewards = []
    
    for episode in range(args.eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nMean reward over {args.eval_episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Dual-arm brachiation simulation with MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive viewer
  python main.py --mode demo              # Run demo simulation
  python main.py --mode train             # Train with PPO
  python main.py --mode eval --model-path ./checkpoints/model.zip
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="viewer",
        choices=["viewer", "demo", "train", "eval"],
        help="Mode to run: viewer (interactive), demo (random actions), train (RL), eval (evaluate model)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization (for servers/training)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments for training (default: 4)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps for training (default: 1,000,000)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model for evaluation"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of demo in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
    
    # Run selected mode
    if args.mode == "viewer":
        if args.headless:
            print("Warning: --headless has no effect in viewer mode")
        run_viewer()
    elif args.mode == "demo":
        run_demo(headless=args.headless, duration=args.duration)
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "eval":
        run_evaluation(args)


if __name__ == "__main__":
    main()