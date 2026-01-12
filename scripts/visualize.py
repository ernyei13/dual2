#!/usr/bin/env python3
"""
Simple visualization script for the dual-arm robot.

This script provides easy-to-use functions to visualize and test the robot
without needing to set up a full training environment.

Usage:
    python scripts/visualize.py           # Interactive viewer
    python scripts/visualize.py --demo    # Run demo with movements
    python scripts/visualize.py --record  # Record a video
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mujoco


def get_model_path() -> str:
    """Get path to the MuJoCo model."""
    model_path = PROJECT_ROOT / "mujoco" / "robot.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return str(model_path)


def interactive_viewer(paused: bool = True):
    """Launch the interactive MuJoCo viewer."""
    import mujoco.viewer
    
    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    
    # Load the hanging keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hanging")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    print("=" * 50)
    print("Interactive MuJoCo Viewer")
    print("=" * 50)
    print("\nControls:")
    print("  Mouse drag: Rotate camera")
    print("  Right-click drag: Pan camera")
    print("  Scroll: Zoom")
    print("  Double-click: Track body")
    print("  Space: Pause/Resume")
    print("  Backspace: Reset")
    print("  ESC: Exit")
    if paused:
        print("\n  >>> Starting PAUSED - Press SPACE to start <<<")
    print("=" * 50)
    
    mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)


def run_demo(duration: float = 10.0):
    """Run a demo with sinusoidal joint movements."""
    import mujoco.viewer
    
    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    
    print(f"Running demo for {duration} seconds...")
    print(f"Robot has {model.nu} actuators")
    
    # Print joint names
    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = data.time
        
        while viewer.is_running() and (data.time - start_time) < duration:
            t = data.time
            
            # Create smooth sinusoidal motion for each joint
            for i in range(model.nu):
                freq = 0.5 + i * 0.1  # Different frequency for each joint
                phase = i * 0.5  # Phase offset
                amplitude = 0.5
                data.ctrl[i] = amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            mujoco.mj_step(model, data)
            viewer.sync()
    
    print("Demo completed!")


def test_joint_limits():
    """Test each joint through its full range of motion."""
    import mujoco.viewer
    
    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    
    print("Testing joint limits...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for joint_idx in range(model.nu):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_idx)
            print(f"\nTesting joint {joint_idx}: {joint_name}")
            
            # Sweep through range
            for phase in np.linspace(0, 2 * np.pi, 200):
                if not viewer.is_running():
                    return
                
                # Set all controls to 0
                data.ctrl[:] = 0
                # Sweep current joint
                data.ctrl[joint_idx] = np.sin(phase)
                
                mujoco.mj_step(model, data)
                viewer.sync()
    
    print("\nJoint limit test completed!")


def record_video(output_path: str = "robot_demo.mp4", duration: float = 5.0, fps: int = 30):
    """Record a video of the robot."""
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio imageio-ffmpeg")
        return
    
    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    # Reduced resolution to match default framebuffer limit
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    frames = []
    n_frames = int(duration * fps)
    sim_steps_per_frame = int(1.0 / (fps * model.opt.timestep))
    
    print(f"Recording {duration}s video at {fps} fps...")
    
    for frame_idx in range(n_frames):
        t = frame_idx / fps
        
        # Apply motion
        for i in range(model.nu):
            data.ctrl[i] = 0.5 * np.sin(2 * np.pi * 0.5 * t + i * 0.5)
        
        # Step simulation
        for _ in range(sim_steps_per_frame):
            mujoco.mj_step(model, data)
        
        # Render frame
        renderer.update_scene(data)
        frames.append(renderer.render())
        
        if frame_idx % fps == 0:
            print(f"  Frame {frame_idx}/{n_frames} ({100*frame_idx/n_frames:.0f}%)")
    
    # Save video
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Done!")


def print_model_info():
    """Print information about the MuJoCo model."""
    model = mujoco.MjModel.from_xml_path(get_model_path())
    data = mujoco.MjData(model)
    
    # Forward kinematics for initial state
    mujoco.mj_forward(model, data)
    
    print("=" * 60)
    print("MuJoCo Model Information")
    print("=" * 60)
    
    print(f"\nGeneral:")
    print(f"  Model name: {model.name if hasattr(model, 'name') else 'N/A'}")
    print(f"  Timestep: {model.opt.timestep} s")
    print(f"  Generalized coordinates (nq): {model.nq}")
    print(f"  Generalized velocities (nv): {model.nv}")
    print(f"  Actuators (nu): {model.nu}")
    print(f"  Bodies (nbody): {model.nbody}")
    print(f"  Joints (njnt): {model.njnt}")
    print(f"  Geometries (ngeom): {model.ngeom}")
    
    print(f"\nBodies:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = model.body_mass[i]
        print(f"  {i}: {name} (mass: {mass:.4f} kg)")
    
    print(f"\nJoints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        limited = "limited" if model.jnt_limited[i] else "unlimited"
        if model.jnt_limited[i]:
            range_str = f"[{model.jnt_range[i, 0]:.2f}, {model.jnt_range[i, 1]:.2f}]"
        else:
            range_str = ""
        print(f"  {i}: {name} ({jnt_type}, {limited}) {range_str}")
    
    print(f"\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")
    
    print(f"\nSensors:")
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        print(f"  {i}: {name}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and test the dual-arm robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="viewer",
        choices=["viewer", "demo", "test", "record", "info"],
        help="Mode: viewer (interactive), demo (movements), test (joint limits), record (video), info (model info)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration for demo/recording in seconds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="robot_demo.mp4",
        help="Output path for video recording"
    )
    
    args = parser.parse_args()
    
    if args.mode == "viewer":
        interactive_viewer()
    elif args.mode == "demo":
        run_demo(duration=args.duration)
    elif args.mode == "test":
        test_joint_limits()
    elif args.mode == "record":
        record_video(output_path=args.output, duration=args.duration)
    elif args.mode == "info":
        print_model_info()


if __name__ == "__main__":
    main()
