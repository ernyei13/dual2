#!/usr/bin/env python3
"""
Find configuration where arm1 gripper grips wall1 top.
Wall1 top is at (0.15, 0, 0.30).
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
    
    # Target: wall1 top
    target = np.array([0.15, 0, 0.30])
    
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'arm1_tip')
    
    best_dist = float('inf')
    best_qpos = None
    
    print("Searching for grip configuration...")
    print(f"Target: {target}")
    
    # Search over base positions and joint angles
    for base_x in np.arange(-0.10, 0.20, 0.02):
        for base_z in np.arange(0.30, 0.50, 0.02):
            for shoulder in np.arange(-2.5, 0.2, 0.1):
                for elbow in np.arange(0, 3.14, 0.1):
                    for wrist_pitch in np.arange(-3.0, 0.1, 0.1):
                        mujoco.mj_resetData(model, data)
                        
                        # Base position and orientation
                        data.qpos[0] = base_x
                        data.qpos[1] = 0
                        data.qpos[2] = base_z
                        data.qpos[3:7] = [1, 0, 0, 0]
                        
                        # Arm1 joints
                        data.qpos[7] = shoulder
                        data.qpos[8] = elbow
                        data.qpos[9] = wrist_pitch
                        data.qpos[10] = 0  # wrist_roll
                        data.qpos[11] = -1.0  # gripper closed
                        
                        # Arm2 joints
                        data.qpos[12] = 0
                        data.qpos[13] = 0.5
                        data.qpos[14] = 0
                        
                        mujoco.mj_forward(model, data)
                        
                        tip_pos = data.site_xpos[tip_id].copy()
                        dist = np.linalg.norm(tip_pos - target)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_qpos = data.qpos.copy()
                            best_tip = tip_pos.copy()
                            print(f"  New best: dist={dist:.4f}, tip={tip_pos}")
    
    print(f"\n=== BEST CONFIGURATION ===")
    print(f"Distance: {best_dist:.4f} m")
    print(f"Tip position: {best_tip}")
    print(f"Target: {target}")
    print(f"\nqpos: {' '.join([f'{q:.3f}' for q in best_qpos])}")
    
    # Format for XML
    base_x, base_y, base_z = best_qpos[0], best_qpos[1], best_qpos[2]
    shoulder, elbow, wrist_pitch = best_qpos[7], best_qpos[8], best_qpos[9]
    
    print(f"\n=== FOR ROBOT.XML ===")
    print(f'Base body: pos="{base_x:.2f} {base_y:.2f} {base_z:.2f}"')
    qpos_str = " ".join([f"{q:.2f}" for q in best_qpos])
    ctrl_str = f"{shoulder:.2f} {elbow:.2f} {wrist_pitch:.2f} 0 -1.0 0 0.5 0"
    print(f'qpos="{qpos_str}"')
    print(f'ctrl="{ctrl_str}"')
    
    # Set the best configuration and visualize
    data.qpos[:] = best_qpos
    mujoco.mj_forward(model, data)
    
    print("\nLaunching viewer with best configuration...")
    print("Adjust with sliders if needed, then Copy state.")
    
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
