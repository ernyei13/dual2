#!/usr/bin/env python3
"""
Interactive viewer with physics paused but controls enabled.
Allows manual positioning of the robot to grip wall1.
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "mujoco" / "robot.xml"

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    
    # Load keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'hanging')
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    mujoco.mj_forward(model, data)
    
    # Get tip position
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'arm1_tip')
    
    print("=" * 50)
    print("Interactive Setup - Full Controls Enabled")
    print("=" * 50)
    print(f"Target: wall1 top at (0.15, 0, 0.30)")
    print(f"Current tip: {data.site_xpos[tip_id]}")
    print()
    print("Instructions:")
    print("  1. Click 'Pause' button to stop physics")
    print("  2. Use Control sliders on right to move joints")
    print("  3. Click 'Copy state' to get qpos values")
    print("  4. Press ESC to exit")
    print("=" * 50)
    
    # Use blocking launch for full UI control
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
