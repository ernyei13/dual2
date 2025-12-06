#!/usr/bin/env python3
"""
Find the exact robot configuration to grip the top of wall1.
Wall1 top is at (0.15, 0, 0.30).
"""

import numpy as np
import mujoco
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "mujoco" / "robot.xml"

def find_grip_configuration():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    
    # Target: top of wall1 at (0.15, 0, 0.30)
    target = np.array([0.15, 0, 0.30])
    
    best_dist = float('inf')
    best_config = None
    
    print("Searching for optimal grip configuration...")
    print(f"Target position: {target}")
    
    # Fine-grained search
    for base_x in np.arange(-0.10, 0.15, 0.005):
        for base_z in np.arange(0.25, 0.45, 0.005):
            for shoulder in np.arange(-0.5, 0.2, 0.02):
                for elbow in np.arange(1.5, 3.14, 0.02):
                    for wrist_pitch in np.arange(-0.8, 0.2, 0.02):
                        mujoco.mj_resetData(model, data)
                        
                        # Set base position
                        data.qpos[0] = base_x
                        data.qpos[1] = 0
                        data.qpos[2] = base_z
                        data.qpos[3:7] = [1, 0, 0, 0]  # upright quaternion
                        
                        # Set arm1 joints
                        data.qpos[7] = shoulder      # arm1_shoulder
                        data.qpos[8] = elbow         # arm1_elbow
                        data.qpos[9] = wrist_pitch   # arm1_wrist_pitch
                        data.qpos[10] = 0            # arm1_wrist_roll
                        data.qpos[11] = -1.0         # arm1_gripper (closed)
                        
                        # arm2 joints
                        data.qpos[12] = 0            # arm2_shoulder
                        data.qpos[13] = 0.5          # arm2_wrist_roll
                        data.qpos[14] = 0            # arm2_gripper
                        
                        mujoco.mj_forward(model, data)
                        
                        # Get tip position
                        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'arm1_tip')
                        tip_pos = data.site_xpos[tip_id].copy()
                        
                        dist = np.linalg.norm(tip_pos - target)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_config = {
                                'base_x': base_x,
                                'base_z': base_z,
                                'shoulder': shoulder,
                                'elbow': elbow,
                                'wrist_pitch': wrist_pitch,
                                'tip_pos': tip_pos,
                                'qpos': data.qpos.copy()
                            }
    
    print(f"\n=== Best Configuration Found ===")
    print(f"Distance to target: {best_dist:.4f} m")
    print(f"Base position: x={best_config['base_x']:.3f}, z={best_config['base_z']:.3f}")
    print(f"Shoulder: {best_config['shoulder']:.3f}")
    print(f"Elbow: {best_config['elbow']:.3f}")
    print(f"Wrist pitch: {best_config['wrist_pitch']:.3f}")
    print(f"Tip position: {best_config['tip_pos']}")
    print(f"Target: {target}")
    
    # Generate the keyframe string
    qpos = best_config['qpos']
    qpos_str = " ".join([f"{q:.4f}" for q in qpos])
    ctrl_str = f"{best_config['shoulder']:.4f} {best_config['elbow']:.4f} {best_config['wrist_pitch']:.4f} 0.0 -1.0 0.0 0.5 0.0"
    
    print(f"\n=== For robot.xml ===")
    print(f'Base pos: pos="{best_config["base_x"]:.3f} 0 {best_config["base_z"]:.3f}"')
    print(f'qpos="{qpos_str}"')
    print(f'ctrl="{ctrl_str}"')
    
    # Update the robot.xml file
    update_robot_xml(best_config)
    
    return best_config

def update_robot_xml(config):
    """Update the robot.xml file with the found configuration."""
    xml_content = MODEL_PATH.read_text()
    
    # Update base position
    import re
    
    # Update base body position
    xml_content = re.sub(
        r'<body name="base" pos="[^"]*">',
        f'<body name="base" pos="{config["base_x"]:.3f} 0 {config["base_z"]:.3f}">',
        xml_content
    )
    
    # Update keyframe
    qpos_str = f'{config["base_x"]:.3f} 0 {config["base_z"]:.3f} 1 0 0 0 {config["shoulder"]:.3f} {config["elbow"]:.3f} {config["wrist_pitch"]:.3f} 0.0 -1.0 0.0 0.5 0.0'
    ctrl_str = f'{config["shoulder"]:.3f} {config["elbow"]:.3f} {config["wrist_pitch"]:.3f} 0.0 -1.0 0.0 0.5 0.0'
    
    # Replace the keyframe section
    keyframe_pattern = r'<keyframe>.*?</keyframe>'
    new_keyframe = f'''<keyframe>
    <key name="hanging" 
         qpos="{qpos_str}"
         ctrl="{ctrl_str}"/>
  </keyframe>'''
    
    xml_content = re.sub(keyframe_pattern, new_keyframe, xml_content, flags=re.DOTALL)
    
    MODEL_PATH.write_text(xml_content)
    print(f"\n✓ Updated {MODEL_PATH}")

def visualize_result():
    """Launch viewer to see the result."""
    import mujoco.viewer
    
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    
    # Load the keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'hanging')
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    
    mujoco.mj_forward(model, data)
    
    # Print current tip position
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'arm1_tip')
    print(f"\nCurrent tip position: {data.site_xpos[tip_id]}")
    print("Target: (0.15, 0, 0.30)")
    
    print("\nLaunching viewer... Press Space to start physics.")
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    config = find_grip_configuration()
    visualize_result()
