#!/usr/bin/env python3
"""
Find joint configuration to grip wall1 while base is fixed at (0.15, 0, 0.30).
"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")

# Wall1 bar center at (0.15, 0, 0.31)
target = np.array([0.15, 0.0, 0.31])

print(f"Target: {target}")
print("Base fixed at: (0.15, 0, 0.30)")

best_dist = float('inf')
best_config = None

# Base is fixed
data.qpos[0:3] = [0.15, 0, 0.30]
data.qpos[3:7] = [1, 0, 0, 0]

# Search joint space
# We need to reach down, so shoulder should probably be pitched down
for shoulder in np.arange(-3.0, 1.0, 0.2):
    for elbow in np.arange(0, 3.14, 0.2):
        for wrist_pitch in np.arange(-3.0, 1.0, 0.2):
            for wrist_roll in np.arange(-1.0, 2.5, 0.2):
                data.qpos[7] = shoulder
                data.qpos[8] = elbow
                data.qpos[9] = wrist_pitch
                data.qpos[10] = wrist_roll
                data.qpos[11] = -0.55  # gripper closed
                data.qpos[12:15] = [0, 0.5, 0]
                
                mujoco.mj_forward(model, data)
                tip_pos = data.site_xpos[arm1_tip_id].copy()
                
                dist = np.linalg.norm(tip_pos - target)
                
                if dist < best_dist:
                    best_dist = dist
                    best_config = {
                        'shoulder': shoulder,
                        'elbow': elbow,
                        'wrist_pitch': wrist_pitch,
                        'wrist_roll': wrist_roll,
                        'tip_pos': tip_pos.copy(),
                    }
                    if dist < 0.02:
                         print(f"Found candidate: dist={dist:.4f}, tip={tip_pos}")

print()
print("=== BEST CONFIGURATION ===")
print(f"Distance: {best_dist:.4f} m")
print(f"Tip position: {best_config['tip_pos']}")
print(f"Shoulder: {best_config['shoulder']:.2f}")
print(f"Elbow: {best_config['elbow']:.2f}")
print(f"Wrist pitch: {best_config['wrist_pitch']:.2f}")
print(f"Wrist roll: {best_config['wrist_roll']:.2f}")
print()

qpos = f"0.15 0 0.30 1 0 0 0 {best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} {best_config['wrist_roll']:.2f} -0.55 0 0.5 0"
ctrl = f"{best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} {best_config['wrist_roll']:.2f} -0.55 0 0.5 0"
print(f'qpos="{qpos}"')
print(f'ctrl="{ctrl}"')
