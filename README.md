## Robot and project aim
The robot is based on the SO-ARM100 robotic arm, modified to incorporate a gripper at each end effector. The project aims to develop a reinforcement learning policy that enables the robot to navigate within a defined space by sequentially grasping objects, thereby achieving locomotion.

![CAD Design](docs/images/cad-design-1.png)

## Running the Training

This project uses Isaac Lab for reinforcement learning. The following instructions explain how to start the training process for the dual-arm brachiation task on a Windows machine.

### Prerequisites

*   You have successfully installed NVIDIA Isaac Lab on your Windows machine.
*   You have cloned this repository.

### Launching the Training

To start the training and visualize the process in the Isaac Sim GUI, follow these steps:

1.  **Open the Isaac Lab Terminal**: Navigate to your Isaac Lab installation directory and run the batch file that opens a pre-configured command prompt. This ensures all necessary environment variables are set.

2.  **Navigate to Project Directory**: In the terminal that just opened, navigate to the root of this project folder.
    ```cmd
    cd path\to\your\project\dual2
    ```

3.  **Run the Training Script**: Execute the following command to launch Isaac Sim, load the environment, and begin training.
    ```cmd
    isaaclab.bat -p src/main.py --task DualArmBrachiation
    ```

This command runs the `src/main.py` script. Because the `--headless` flag is not specified, Isaac Sim will launch with its full graphical user interface, allowing you to watch the robot learn in real-time.

### Customizing the Training Run

You can modify the training parameters directly from the command line. For example, to run with a different number of parallel environments:

```cmd
isaaclab.bat -p src/main.py --task DualArmBrachiation --num_envs 1024
