#!/usr/bin/env python3
"""
Interactive script to manually find the correct grip configuration.
Shows the arm1_tip position in real-time.
"""
import mujoco
import mujoco.viewer
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

# Get site ID
arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")

print("=== Grip Configuration Finder ===")
print("Target position: wall1 top at (0.15, 0, 0.30)")
print()

# Keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

tip_pos = data.site_xpos[arm1_tip_id].copy()
print(f"Current arm1_tip position: {tip_pos}")
print(f"Distance to target: {np.linalg.norm(tip_pos - np.array([0.15, 0, 0.30])):.4f}")

print()
print("Let's try different base positions:")
print()

# Test different positions
best_dist = float('inf')
best_config = None

# The problem is the arm extends to y+ direction due to its mounting
# We need to rotate the base so arm1 points toward +x
# Or position base at negative y so the arm can reach

for base_x in np.arange(-0.15, 0.15, 0.02):
    for base_y in np.arange(-0.15, 0.15, 0.02):
        for base_z in np.arange(0.25, 0.45, 0.02):
            # Test with different base quaternions to rotate around z
            for yaw in np.arange(-np.pi, np.pi, np.pi/6):
                quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])  # Rotation around z
                
                for shoulder in np.arange(-2.9, 0.2, 0.3):
                    for elbow in np.arange(0, 3.14, 0.3):
                        for wrist_pitch in np.arange(-3, 0.1, 0.3):
                            # Set base pose
                            data.qpos[0:3] = [base_x, base_y, base_z]
                            data.qpos[3:7] = quat
                            # Set arm1 joints
                            data.qpos[7] = shoulder
                            data.qpos[8] = elbow
                            data.qpos[9] = wrist_pitch
                            data.qpos[10] = 0  # wrist_roll
                            data.qpos[11] = -0.8  # gripper closed around bar
                            # Arm2 neutral
                            data.qpos[12:15] = [0, 0.5, 0]
                            
                            mujoco.mj_forward(model, data)
                            tip_pos = data.site_xpos[arm1_tip_id].copy()
                            
                            target = np.array([0.15, 0, 0.31])
                            dist = np.linalg.norm(tip_pos - target)
                            
                            if dist < best_dist:
                                best_dist = dist
                                best_config = {
                                    'base_pos': [base_x, base_y, base_z],
                                    'quat': quat.copy(),
                                    'shoulder': shoulder,
                                    'elbow': elbow,
                                    'wrist_pitch': wrist_pitch,
                                    'tip_pos': tip_pos.copy()
                                }
                                print(f"New best: dist={dist:.4f}, tip={tip_pos}, base=({base_x:.2f},{base_y:.2f},{base_z:.2f}), yaw={np.degrees(yaw):.0f}°")

print()
print("=== BEST CONFIGURATION ===")
print(f"Distance: {best_dist:.4f}")
print(f"Base pos: {best_config['base_pos']}")
print(f"Base quat: {best_config['quat']}")  
print(f"Shoulder: {best_config['shoulder']:.2f}")
print(f"Elbow: {best_config['elbow']:.2f}")
print(f"Wrist pitch: {best_config['wrist_pitch']:.2f}")
print(f"Tip position: {best_config['tip_pos']}")
print()
print("Keyframe qpos format:")
qpos_str = " ".join([
    f"{best_config['base_pos'][0]:.3f}",
    f"{best_config['base_pos'][1]:.3f}",
    f"{best_config['base_pos'][2]:.3f}",
    f"{best_config['quat'][0]:.4f}",
    f"{best_config['quat'][1]:.4f}",
    f"{best_config['quat'][2]:.4f}",
    f"{best_config['quat'][3]:.4f}",
    f"{best_config['shoulder']:.2f}",
    f"{best_config['elbow']:.2f}",
    f"{best_config['wrist_pitch']:.2f}",
    "0",  # wrist_roll
    "-0.8",  # gripper
    "0", "0.5", "0"  # arm2
])
print(f'qpos="{qpos_str}"')
