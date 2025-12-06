#!/usr/bin/env python3
"""
Refine the grip configuration using optimization to get exactly to the target.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

arm1_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm1_tip")
target = np.array([0.15, 0.0, 0.31])

print(f"Target: {target}")

# Initial guess from previous search
# Shoulder: 0.40, Elbow: 2.80, Wrist pitch: -0.20, Wrist roll: -1.00
# Add Yaw: 0
x0 = np.array([0.0, 0.2, 2.80, -0.20, -1.00])

def objective(x):
    # Set joints
    yaw = x[0]
    quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    
    data.qpos[0:3] = [0.15, 0, 0.30]
    data.qpos[3:7] = quat
    data.qpos[7] = x[1] # shoulder
    data.qpos[8] = x[2] # elbow
    data.qpos[9] = x[3] # wrist_pitch
    data.qpos[10] = x[4] # wrist_roll
    data.qpos[11] = -0.55
    data.qpos[12:15] = [0, 0.5, 0]
    
    mujoco.mj_forward(model, data)
    tip_pos = data.site_xpos[arm1_tip_id]
    
    return np.linalg.norm(tip_pos - target)

# Bounds for joints (relaxed to find feasible solution)
bounds = [
    (-3.14, 3.14), # yaw
    (-3.14, 3.14), # shoulder
    (-3.14, 3.14), # elbow
    (-3.14, 3.14), # wrist_pitch
    (-3.14, 3.14)  # wrist_roll
]

# Try multiple initial guesses
guesses = [
    [0.0, 0.2, 2.80, -0.20, -1.00],
    [0.0, -1.0, 1.5, -1.0, 0.0],
    [3.14, -1.0, 1.5, -1.0, 0.0],
    [1.57, -1.0, 1.5, -1.0, 0.0],
    [-1.57, -1.0, 1.5, -1.0, 0.0],
    [0.0, 0.0, 2.0, 0.0, 0.0],
    [0.0, -0.5, 2.5, -0.5, 0.5],
    [0.0, 0.5, 1.0, 0.5, -0.5],
    [1.57, 0.0, 2.0, 0.0, 1.0],
    [-1.57, 0.0, 2.0, 0.0, -1.0],
]

best_res = None
best_dist = float('inf')

print("Optimizing with multiple guesses...")

for x0 in guesses:
    res = minimize(objective, np.array(x0), bounds=bounds, method='L-BFGS-B')
    print(f"Guess {x0} -> dist={res.fun:.6f}, success={res.success}")
    print(f"  Config: Yaw={res.x[0]:.4f}, Joints={res.x[1:]}")
    if res.fun < best_dist:
        best_dist = res.fun
        best_res = res

print(f"Best distance: {best_dist:.6f} m")
print(f"Best config: Yaw={best_res.x[0]:.4f}, Joints={best_res.x[1:]}")

res = best_res


# Verify
yaw = res.x[0]
quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
data.qpos[3:7] = quat
data.qpos[7:11] = res.x[1:]
mujoco.mj_forward(model, data)
tip_pos = data.site_xpos[arm1_tip_id]
print(f"Final tip pos: {tip_pos}")

# Generate XML string
q = quat
qpos = f"0.15 0 0.30 {q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f} {res.x[1]:.4f} {res.x[2]:.4f} {res.x[3]:.4f} {res.x[4]:.4f} -0.55 0 0.5 0"
ctrl = f"{res.x[1]:.4f} {res.x[2]:.4f} {res.x[3]:.4f} {res.x[4]:.4f} -0.55 0 0.5 0"


print("\n=== FOR robot.xml ===")
print(f'qpos="{qpos}"')
print(f'ctrl="{ctrl}"')
