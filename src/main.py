# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES, All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Main entry-point for training the dual-arm brachiation robot.
"""

import argparse
from typing import Type

# convenient way to get Isaac Lab modules
from omni.isaac.lab.app import AppLauncher

# import the environment and runner configurations
from environments.dual_arm_walk_env_cfg import DualArmWalkEnvCfg
from workflows.ppo_runner_cfg import BrachiationPPORunnerCfg

# import the RLEnv and runner classes
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_rl.rl_games import RlGamesGpuPPORunner

# utility to parse CLI arguments
from omni.isaac.lab_tasks.utils import parse_cfg_cli


def main(args: argparse.Namespace):
    """Main function to create and train the brachiation environment."""
    # 1. Parse configuration and command-line arguments
    # This utility function from Isaac Lab handles merging of config files and CLI arguments
    env_cfg_class: Type[DualArmWalkEnvCfg] = DualArmWalkEnvCfg
    runner_cfg_class: Type[BrachiationPPORunnerCfg] = BrachiationPPORunnerCfg

    env_cfg: DualArmWalkEnvCfg = parse_cfg_cli(
        task_name=args.task,
        cfg_cls=env_cfg_class,
        config_file=None, # Not using a separate config file for env
    )
    runner_cfg: BrachiationPPORunnerCfg = runner_cfg_class(
        experiment_name=args.experiment_name, run_name=args.run_name, resume=args.resume
    )

    # 2. Create the RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3. Create the RL runner
    runner = RlGamesGpuPPORunner(cfg=runner_cfg)

    # 4. Assign the environment to the runner and start training or playing
    if args.play:
        print(f"[INFO]: Loading model from: {runner_cfg.load_path}")
        runner.play(env)
    else:
        runner.learn(env)

    # 5. Cleanly close the environment and simulation
    env.close()


if __name__ == "__main__":
    # 1. Create the argument parser
    parser = argparse.ArgumentParser(description="Train a dual-gripper robot for brachiation.")

    # 2. Add arguments for AppLauncher and script
    parser.add_argument("--headless", action="store_true", default=False, help="Force headless mode for the simulation.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument("--task", type=str, default="DualArmWalk", help="Name of the task to run.")
    parser.add_argument("--play", action="store_true", default=False, help="Run in play mode (inference).")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
    parser.add_argument("--experiment_name", type=str, default="DualArm_Brachiation", help="Name of the experiment for logging.")
    parser.add_argument("--run_name", type=str, default="", help="Name of the run for logging. Empty for auto-generated.")

    # 3. Parse arguments and launch the application
    # The `parse_cfg_cli` utility will add its own arguments for config overriding, so we use `parse_known_args`
    args, unknown_args = parser.parse_known_args()

    # 4. Launch the simulation and run the main function
    app_launcher = AppLauncher(headless=args.headless)
    # Pass the parsed arguments to the main function
    app_launcher.launch(lambda: main(args))