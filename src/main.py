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
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    print("Starting PPO training (Improved)...")
    
    # Create environment factory
    def make_env():
        return BrachiationEnv(
            render_mode="rgb_array" if args.headless else None,
            initial_keyframe="wall1_grip",
            task_mode=args.task,
            curriculum_level=args.curriculum_level,
        )
    
    # Create vectorized environment
    # Using SubprocVecEnv for parallel execution is good, but let's ensure it's robust
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env])
        
    # IMPORTANT: Normalize observations and rewards for PPO stability
    # Clip observations to 10.0
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.995,
    )
    
    # ADVANCED IMPROVEMENT: Frame Stacking
    # Stack 4 frames to give temporal context (acceleration, jerk)
    env = VecFrameStack(env, n_stack=4)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // args.num_envs, 1), # Save every X steps per env
        save_path="./checkpoints/",
        name_prefix="brachiation_ppo"
    )
    
    # Linear learning rate schedule
    def linear_schedule(initial_value: float):
        """
        Linear learning rate schedule.
        :param initial_value: Initial learning rate.
        :return: schedule function.
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value
        return func

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(3e-4), # Decay to 0
        n_steps=4096, # Longer horizon for longer episodes
        batch_size=1024, # Larger batch size for more stable gradients
        n_epochs=10,
        gamma=0.995, # Higher gamma for longer horizon
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005, # Reduce entropy slightly as training progresses
        tensorboard_log="./logs/tensorboard/",
        device="auto",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), # Explicit network size
    )
    
    print(f"Training for {args.total_timesteps} timesteps with {args.num_envs} environments...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # Save the normalized environment stats so evaluation works correctly!
    env.save("./checkpoints/vec_normalize.pkl")
    
    # Save final model
    model.save("./checkpoints/brachiation_final")
    print("Training completed! Model saved to ./checkpoints/brachiation_final")
    
    env.close()


def run_evaluation(args):
    """Evaluate a trained model."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
    from stable_baselines3 import PPO
    from src.envs.brachiation_env import BrachiationEnv
    import numpy as np
    
    model_path = args.model_path or "./checkpoints/brachiation_final"
    vec_norm_path = "./checkpoints/vec_normalize.pkl"
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Determine render mode
    render_mode = "human"
    if args.headless or args.record_video:
        render_mode = "rgb_array"
        
    # We must wrap the env in VecNormalize and load stats to match training distribution
    # We use a DummyVecEnv for evaluation
    env = DummyVecEnv([lambda: BrachiationEnv(
        render_mode=render_mode,
        initial_keyframe="wall1_grip",
        task_mode=args.task,
    )])
    
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        # Don't update stats during evaluation
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: VecNormalize stats not found. Evaluation might be poor.")

    # Also need FrameStack for evaluation if used in training
    # Must be applied AFTER normalization (on top of it) to match training stack:
    # Env -> Normalize -> FrameStack
    env = VecFrameStack(env, n_stack=4)
    
    print("Running evaluation...")
    total_rewards = []
    
    # Video recording setup
    frames = []
    record_video = args.record_video
    
    for episode in range(args.eval_episodes):
        obs = env.reset() # VecEnv reset returns just obs
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # VecEnv step returns (obs, reward, done, info) - vectorized!
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Render frame if recording
            if record_video and len(frames) < 1500: # Limit to ~30s at 50fps
                # env.render() returns the frame in rgb_array mode
                frames.append(env.render())
                
            # Done is an array of booleans in VecEnv
            if done[0]:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nMean reward over {args.eval_episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()
    
    if record_video and len(frames) > 0:
        import imageio
        output_path = "trained_model_demo.mp4"
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=50)
        print("Video saved!")


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
    
    parser.add_argument("--record-video", action="store_true", help="Record video during evaluation")
    parser.add_argument("--task", type=str, default="traversal", choices=["traversal", "grasping"], help="Task to train (traversal or grasping)")
    parser.add_argument("--curriculum-level", type=int, default=8, help="Starting wall for curriculum (0-9). Higher = easier.")
    
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