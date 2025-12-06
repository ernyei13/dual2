#!/usr/bin/env python3
"""
Find where arm1_tip can naturally reach, then we'll move wall there.
"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")

print("Finding natural grip positions for arm1...")
print("Target height: z ≈ 0.30 (wall top)")
print()

best_dist = float('inf')
best_config = None

# Fixed base position
base_x, base_y, base_z = 0.05, 0, 0.35

# Search arm joint space
for shoulder in np.arange(-2.9, 0.2, 0.1):
    for elbow in np.arange(0, 3.14, 0.1):
        for wrist_pitch in np.arange(-3, 0.1, 0.1):
            for wrist_roll in np.arange(-0.8, 2.3, 0.3):
                # Set base pose (identity quat)
                data.qpos[0:3] = [base_x, base_y, base_z]
                data.qpos[3:7] = [1, 0, 0, 0]
                # Set arm1 joints
                data.qpos[7] = shoulder
                data.qpos[8] = elbow
                data.qpos[9] = wrist_pitch
                data.qpos[10] = wrist_roll
                data.qpos[11] = -0.8  # gripper closed
                data.qpos[12:15] = [0, 0.5, 0]
                
                mujoco.mj_forward(model, data)
                tip_pos = data.site_xpos[arm1_tip_id].copy()
                
                # We want: x as far forward as possible, y close to 0, z around 0.30
                # Score: maximize x, minimize |y|, minimize |z - 0.30|
                
                z_target = 0.30
                z_err = abs(tip_pos[2] - z_target)
                y_err = abs(tip_pos[1])
                
                # Only consider positions where z is close to 0.30 and y is close to 0
                if z_err < 0.02 and y_err < 0.03 and tip_pos[0] > 0.08:
                    score = tip_pos[0] - 5*y_err - 5*z_err  # maximize x, minimize errors
                    
                    if best_config is None or score > best_config.get('score', -999):
                        best_config = {
                            'score': score,
                            'tip_pos': tip_pos.copy(),
                            'shoulder': shoulder,
                            'elbow': elbow,
                            'wrist_pitch': wrist_pitch,
                            'wrist_roll': wrist_roll,
                        }
                        print(f"Good pos: x={tip_pos[0]:.3f}, y={tip_pos[1]:.3f}, z={tip_pos[2]:.3f}, "
                              f"shoulder={shoulder:.2f}, elbow={elbow:.2f}, wpitch={wrist_pitch:.2f}, wroll={wrist_roll:.2f}")

if best_config:
    print()
    print("=== BEST CONFIGURATION ===")
    print(f"Tip position: {best_config['tip_pos']}")
    print(f"Shoulder: {best_config['shoulder']:.2f}")
    print(f"Elbow: {best_config['elbow']:.2f}")
    print(f"Wrist pitch: {best_config['wrist_pitch']:.2f}")
    print(f"Wrist roll: {best_config['wrist_roll']:.2f}")
    print()
    print("Suggested wall position (put wall where arm reaches):")
    print(f'  wall1 pos="{best_config["tip_pos"][0]:.3f} 0 0.15"')
    print()
    print("Keyframe qpos:")
    qpos_str = f"0.05 0 0.35 1 0 0 0 {best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} {best_config['wrist_roll']:.2f} -0.8 0 0.5 0"
    print(f'qpos="{qpos_str}"')
    ctrl_str = f"{best_config['shoulder']:.2f} {best_config['elbow']:.2f} {best_config['wrist_pitch']:.2f} {best_config['wrist_roll']:.2f} -0.8 0 0.5 0"
    print(f'ctrl="{ctrl_str}"')
else:
    print("No good configurations found - arm may not reach z=0.30 with y≈0")
    print()
    print("Finding best overall positions...")
    
    # Find best overall
    for shoulder in np.arange(-2.9, 0.2, 0.2):
        for elbow in np.arange(0, 3.14, 0.2):
            for wrist_pitch in np.arange(-3, 0.1, 0.2):
                data.qpos[0:3] = [base_x, base_y, base_z]
                data.qpos[3:7] = [1, 0, 0, 0]
                data.qpos[7] = shoulder
                data.qpos[8] = elbow
                data.qpos[9] = wrist_pitch
                data.qpos[10] = 0
                data.qpos[11] = -0.8
                data.qpos[12:15] = [0, 0.5, 0]
                
                mujoco.mj_forward(model, data)
                tip_pos = data.site_xpos[arm1_tip_id].copy()
                
                if tip_pos[0] > 0.1 and abs(tip_pos[2] - 0.30) < 0.03:
                    print(f"Candidate: x={tip_pos[0]:.3f}, y={tip_pos[1]:.3f}, z={tip_pos[2]:.3f}")
