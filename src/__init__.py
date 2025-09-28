# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES, All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

import os
import toml

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import import_module_from_file

# Get the directory of the current file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# Define the path to the configuration file
__config_file__ = os.path.join(__dir__, "config", "brachiation.toml")

# Create a dictionary to store the registered environments
REGISTERED_ENVS = {"DualArmBrachiation": toml.load(__config_file__)["env"]}