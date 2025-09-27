# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES, All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PPO runner configuration for the dual-arm walking (brachiation) task.
"""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_rl.rl_games import RlGamesGpuPPORunnerCfg


@configclass
class BrachiationPPORunnerCfg(RlGamesGpuPPORunnerCfg):
    """PPO runner configuration for the dual-arm brachiation task."""

    # Configuration for the policy network (actor-critic)
    @configclass
    class PolicyCfg(RlGamesGpuPPORunnerCfg.PolicyCfg):
        """Configuration for the policy network."""
        # Use a larger network for this complex task
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        init_noise_std = 1.0

    # Configuration for the PPO algorithm
    @configclass
    class AlgorithmCfg(RlGamesGpuPPORunnerCfg.AlgorithmCfg):
        """Configuration for the PPO algorithm."""
        # PPO-specific hyperparameters
        max_epochs = 5000  # Train for more epochs
        num_steps_per_env = 32
        learning_rate = 3e-4
        gamma = 0.99
        lam = 0.95
        entropy_coef = 0.005 # Small entropy bonus to encourage exploration
        clip_param = 0.2
        grad_norm = 1.0
        num_minibatches = 8

    # Update the runner's config with our custom settings
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
