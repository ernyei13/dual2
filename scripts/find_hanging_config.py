#!/usr/bin/env python3
"""
Find configuration where gripper is ON the bar and robot hangs from it.
The robot swings from grippers - so we need to find base position that 
puts the gripper tip exactly on the wall bar.
"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")

# Wall1 bar is at (0.15, 0, 0.31) - center of the bar
# The bar is a cylinder with radius 0.012, extending in Y direction
bar_center = np.array([0.15, 0.0, 0.31])

print("=== Finding Hanging Configuration ===")
print(f"Bar center: {bar_center}")
print()
print("The robot hangs from arm1_gripper gripping the bar.")
print("We need to find base position such that arm1_tip reaches the bar.")
print()

best_dist = float('inf')
best_config = None

# Search over base position and arm joints
# The robot will hang, so base should be BELOW the bar
for base_x in np.arange(-0.2, 0.2, 0.01):
    for base_y in np.arange(-0.15, 0.15, 0.02):
        for base_z in np.arange(0.05, 0.35, 0.02):
            # Try different arm configurations
            for shoulder in np.arange(-2.9, 0.2, 0.2):
                for elbow in np.arange(0, 3.14, 0.2):
                    for wrist_pitch in np.arange(-3, 0.1, 0.2):
                        for wrist_roll in np.arange(-0.8, 2.3, 0.4):
                            data.qpos[0:3] = [base_x, base_y, base_z]
                            data.qpos[3:7] = [1, 0, 0, 0]  # upright orientation
                            data.qpos[7] = shoulder
                            data.qpos[8] = elbow
                            data.qpos[9] = wrist_pitch
                            data.qpos[10] = wrist_roll
                            data.qpos[11] = -0.8  # gripper mostly closed
                            data.qpos[12:15] = [0, 0.5, 0]  # arm2 neutral
                            
                            mujoco.mj_forward(model, data)
                            tip_pos = data.site_xpos[arm1_tip_id].copy()
                            
                            dist = np.linalg.norm(tip_pos - bar_center)
                            
                            if dist < best_dist:
                                best_dist = dist
                                best_config = {
                                    'base_pos': [base_x, base_y, base_z],
                                    'shoulder': shoulder,
                                    'elbow': elbow,
                                    'wrist_pitch': wrist_pitch,
                                    'wrist_roll': wrist_roll,
                                    'tip_pos': tip_pos.copy(),
                                }
                                if dist < 0.02:
                                    print(f"dist={dist:.4f}, tip={tip_pos}, base=({base_x:.2f},{base_y:.2f},{base_z:.2f})")

print()
print("=== BEST CONFIGURATION ===")
print(f"Distance: {best_dist:.4f} m")
print(f"Tip position: {best_config['tip_pos']}")
print(f"Base position: {best_config['base_pos']}")
print(f"Shoulder: {best_config['shoulder']:.2f}")
print(f"Elbow: {best_config['elbow']:.2f}")
print(f"Wrist pitch: {best_config['wrist_pitch']:.2f}")
print(f"Wrist roll: {best_config['wrist_roll']:.2f}")
print()

# Generate XML config
bx, by, bz = best_config['base_pos']
print("=== FOR robot.xml ===")
print()
print(f'<body name="base" pos="{bx:.3f} {by:.3f} {bz:.3f}">')
print()
print("Keyframe:")
qpos = f"{bx:.3f} {by:.3f} {bz:.3f} 1 0 0 0 {best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} {best_config['wrist_roll']:.2f} -0.8 0 0.5 0"
ctrl = f"{best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} {best_config['wrist_roll']:.2f} -0.8 0 0.5 0"
print(f'<key name="hanging" qpos="{qpos}" ctrl="{ctrl}"/>')
