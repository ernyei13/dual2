import pybullet as p
import pybullet_data
import os
import time

# Start the simulation client
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load the ground plane
p.loadURDF("plane.urdf")

# Load the robot model
robot_start_position = [0, 0, 0.5]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Construct the full path to the URDF file
urdf_path = os.path.join(os.path.dirname(__file__), "robot.urdf")

# Load the URDF from its path
robotId = p.loadURDF(urdf_path, robot_start_position, robot_start_orientation)

# Create joint control sliders
num_joints = p.getNumJoints(robotId)
joint_sliders = []
movable_joints = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
        movable_joints.append(i)
        slider = p.addUserDebugParameter(joint_name, joint_info[8], joint_info[9], 0)
        joint_sliders.append(slider)

# Keep the simulation running
while p.isConnected():
    # Read slider values and set joint positions
    for i in range(len(joint_sliders)):
        slider_value = p.readUserDebugParameter(joint_sliders[i])
        p.setJointMotorControl2(robotId, movable_joints[i], p.POSITION_CONTROL, targetPosition=slider_value)

    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect after the loop ends
p.disconnect()