#!/usr/bin/env python3
"""
Verify the grip configuration by loading the model and checking the keyframe.
"""
import mujoco
import numpy as np
import time

def verify():
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    print("Resetting to 'hanging' keyframe...")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hanging")
    if key_id == -1:
        print("Error: Keyframe 'hanging' not found!")
        return
        
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    
    # Check position
    arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")
    target = np.array([0.15, 0.0, 0.31])
    
    tip_pos = data.site_xpos[arm1_tip_id]
    dist = np.linalg.norm(tip_pos - target)
    
    print(f"Initial State:")
    print(f"  Tip Pos: {tip_pos}")
    print(f"  Target:  {target}")
    print(f"  Error:   {dist:.6f} m")
    
    print("\nStepping simulation for 100 steps (0.2s)...")
    # Enable gravity compensation or hold position?
    # The keyframe sets ctrl to the same values as qpos (for position control)
    # So the actuators should hold the position.
    
    for _ in range(100):
        mujoco.mj_step(model, data)
        
    tip_pos_final = data.site_xpos[arm1_tip_id]
    dist_final = np.linalg.norm(tip_pos_final - target)
    
    print(f"Final State:")
    print(f"  Tip Pos: {tip_pos_final}")
    print(f"  Error:   {dist_final:.6f} m")
    
    if dist_final < 0.01:
        print("\nSUCCESS: Grip is stable and accurate.")
    else:
        print("\nWARNING: Grip drifted or is inaccurate.")

if __name__ == "__main__":
    verify()
