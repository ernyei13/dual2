## Dual-Arm Brachiation RL Project

This project trains a dual-arm robot to move by grasping objects using reinforcement learning in Isaac Lab.

![CAD Design](docs/images/cad-design-1.png)

### Prerequisites

- Isaac Lab (latest version): https://github.com/NVIDIA-Omniverse/IsaacLab
- Clone this repository

### How to Run

1. **Open Isaac Lab Terminal**
    - Launch the Isaac Lab terminal (see Isaac Lab docs for your OS).

2. **Navigate to Project Directory**
    ```powershell
    cd C:\Users\User\Documents\dev\robotics\dual2
    ```

3. **Run Training Script**
    ```powershell
  ..\IsaacLab\isaaclab.bat -p src/main.py --task DualArmBrachiation
    ```
    This launches Isaac Sim with GUI and starts RL training.

#### Options

- Change number of environments:
  ```powershell
  ..\IsaacLab\isaaclab.bat -p src/main.py --task DualArmBrachiation --num_envs 1024
  ```
- Run headless (no GUI):
  ```powershell
  ..\IsaacLab\isaaclab.bat -p src/main.py --task DualArmBrachiation --headless
  ```

### Key Files

- Robot URDF: `assets/urdf/robot.urdf`
- Main script: `src/main.py`
- Environment config: `src/brachiation_env_cfg.py`
- RL config: `src/workflows/ppo_runner_cfg.py`
