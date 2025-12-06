#!/usr/bin/env python3
"""
Find configuration with base rotation to reach wall at (0.15, 0, 0.30).
"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")

target = np.array([0.15, 0.0, 0.30])

best_dist = float('inf')
best_config = None

print("Searching with base rotation...")
print(f"Target: {target}")
print()

# Rotate base around z-axis (yaw) to point arm toward wall
for yaw in np.arange(-np.pi, np.pi, 0.1):
    # Quaternion for rotation around z-axis
    quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    
    for base_x in np.arange(-0.05, 0.10, 0.02):
        for base_z in np.arange(0.28, 0.40, 0.02):
            for shoulder in np.arange(-2.9, 0.2, 0.2):
                for elbow in np.arange(0, 3.14, 0.2):
                    for wrist_pitch in np.arange(-3, 0.1, 0.2):
                        data.qpos[0:3] = [base_x, 0, base_z]
                        data.qpos[3:7] = quat
                        data.qpos[7] = shoulder
                        data.qpos[8] = elbow
                        data.qpos[9] = wrist_pitch
                        data.qpos[10] = 0
                        data.qpos[11] = -0.8
                        data.qpos[12:15] = [0, 0.5, 0]
                        
                        mujoco.mj_forward(model, data)
                        tip_pos = data.site_xpos[arm1_tip_id].copy()
                        
                        dist = np.linalg.norm(tip_pos - target)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_config = {
                                'base_pos': [base_x, 0, base_z],
                                'yaw': yaw,
                                'quat': quat.copy(),
                                'shoulder': shoulder,
                                'elbow': elbow,
                                'wrist_pitch': wrist_pitch,
                                'tip_pos': tip_pos.copy(),
                            }
                            if dist < 0.03:
                                print(f"dist={dist:.4f}, tip={tip_pos}, yaw={np.degrees(yaw):.0f}°, "
                                      f"base=({base_x:.2f}, 0, {base_z:.2f})")

print()
print("=== BEST CONFIGURATION ===")
print(f"Distance to target: {best_dist:.4f}")
print(f"Tip position: {best_config['tip_pos']}")
print(f"Base pos: {best_config['base_pos']}")
print(f"Yaw: {np.degrees(best_config['yaw']):.1f}°")
print(f"Quat: {best_config['quat']}")
print(f"Shoulder: {best_config['shoulder']:.2f}")
print(f"Elbow: {best_config['elbow']:.2f}")
print(f"Wrist pitch: {best_config['wrist_pitch']:.2f}")
print()

# Format for XML
bx, by, bz = best_config['base_pos']
q = best_config['quat']
print("=== FOR robot.xml ===")
print()
print(f'Base body: pos="{bx:.3f} {by:.3f} {bz:.3f}"')
print()
print("Keyframe:")
qpos = f"{bx:.3f} {by:.3f} {bz:.3f} {q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f} {best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} 0 -0.8 0 0.5 0"
ctrl = f"{best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} 0 -0.8 0 0.5 0"
print(f'qpos="{qpos}"')
print(f'ctrl="{ctrl}"')
