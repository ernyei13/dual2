"""
Dual-arm brachiation project using MuJoCo.

This package provides:
- MuJoCo-based simulation environment
- Gymnasium-compatible RL interface
- Training and evaluation utilities
"""

from src.envs import BrachiationEnv, register_env

__all__ = ["BrachiationEnv", "register_env"]
__version__ = "2.0.0"
