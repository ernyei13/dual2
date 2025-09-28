
"""
Main script to run the dual-arm brachiation training.

This script launches the Isaac Lab application and starts the RL training.
"""

import argparse
import os

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Dual-arm brachiation training script.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DualArmBrachiation", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator.")
# append AppLauncher defined arguments
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab_tasks.utils.parse_cfg as parse_cfg
from omni.isaac.lab.utils.dict import print_dict

from src.workflows.ppo_runner_cfg import BrachiationPPORunnerCfg
from omni.isaac.lab_rl.runners import RlGamesGpuPPORunner


def main():
    """Main function."""
    # create the runner config
    runner_cfg: BrachiationPPORunnerCfg = parse_cfg.parse_env_cfg(
        args_cli.task,
        cfg_entry="env",
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_rlg_config=True,
    )
    # print the configuration
    print_dict(runner_cfg)
    # create runner
    runner = RlGamesGpuPPORunner(cfg=runner_cfg)
    # start training
    runner.learn()

if __name__ == "__main__":
    main()