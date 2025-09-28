0,0 +1,41 @@
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES, All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg

##
# Scene definition
##

@configclass
class BrachiationSceneCfg:
    """Configuration for the brachiation scene."""
    # TODO: Add robot and any objects (e.g., bars to grasp)
    pass

##
# Environment configuration
##

@configclass
class BrachiationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dual-arm brachiation environment."""

    # Scene settings
    scene: BrachiationSceneCfg = BrachiationSceneCfg()

    # Basic settings
    episode_length_s = 20.0
    decimation = 2

    # TODO: Add reward functions
    # TODO: Add termination conditions
    # TODO: Add curriculum settings if any

    def __post_init__(self):
        """Post-initialization checks."""
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 8.0, 3.0)