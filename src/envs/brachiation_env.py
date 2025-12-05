"""
Gymnasium-compatible environment for dual-arm brachiation using MuJoCo.

This environment implements a dual-arm robot that learns to move by
grasping and swinging from bars (brachiation locomotion).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

import mujoco


class BrachiationEnv(gym.Env):
    """
    Dual-arm brachiation environment.
    
    The robot must learn to move forward by grasping bars and swinging,
    similar to how gibbons move through trees.
    
    Observation Space:
        - Base position and orientation (7D: xyz + quaternion)
        - Base linear and angular velocity (6D)
        - Joint positions (8D: 5 for arm1, 3 for arm2)
        - Joint velocities (8D)
        - Fingertip positions (6D: 3 for each arm)
        - Target bar position (3D)
        Total: 38D
    
    Action Space:
        - Joint position targets for all actuators (8D)
        Continuous, normalized to [-1, 1]
    
    Rewards:
        - Forward progress reward
        - Height maintenance reward
        - Energy efficiency penalty
        - Grasping bonus
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        control_freq: int = 50,
    ):
        """
        Initialize the brachiation environment.
        
        Args:
            render_mode: "human" for interactive viewer, "rgb_array" for image output
            max_episode_steps: Maximum steps per episode
            control_freq: Control frequency in Hz
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        
        # Load MuJoCo model
        model_path = Path(__file__).parent.parent.parent / "mujoco" / "robot.xml"
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = int(1.0 / (self.control_freq * self.dt))
        
        # Joint and actuator info
        self.n_actuators = self.model.nu
        self.n_joints = self.model.nq
        
        # Get joint limits for action scaling
        self.joint_ranges = self._get_joint_ranges()
        
        # Define observation and action spaces
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actuators,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.initial_base_pos = None
        
        # Viewer for rendering
        self.viewer = None
        self.renderer = None
        
        # Target bar tracking
        self.current_target_bar = 0
        self.bar_positions = self._get_bar_positions()
        
    def _get_joint_ranges(self) -> np.ndarray:
        """Get joint position limits."""
        ranges = []
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                ranges.append(self.model.jnt_range[i])
            else:
                ranges.append(np.array([-np.pi, np.pi]))
        return np.array(ranges)
    
    def _get_bar_positions(self) -> np.ndarray:
        """Get positions of grasping bars."""
        bar_positions = []
        for name in ["bar1", "bar2", "bar3"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                bar_positions.append(self.model.body_pos[body_id].copy())
        return np.array(bar_positions) if bar_positions else np.zeros((3, 3))
    
    def _get_obs_dim(self) -> int:
        """Calculate observation dimension."""
        # Base pose: 3 (pos) + 4 (quat) = 7
        # Base velocity: 3 (linear) + 3 (angular) = 6
        # Joint positions: 8 (actuated joints)
        # Joint velocities: 8
        # Fingertip positions: 6 (2 arms * 3)
        # Target bar position: 3
        return 7 + 6 + 8 + 8 + 6 + 3
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Base position and orientation
        base_pos = self.data.qpos[:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        
        # Base velocities
        base_linvel = self.data.qvel[:3].copy()
        base_angvel = self.data.qvel[3:6].copy()
        
        # Joint positions and velocities (skip freejoint dofs)
        joint_pos = self.data.qpos[7:7+self.n_actuators].copy() if self.n_actuators <= len(self.data.qpos) - 7 else np.zeros(self.n_actuators)
        joint_vel = self.data.qvel[6:6+self.n_actuators].copy() if self.n_actuators <= len(self.data.qvel) - 6 else np.zeros(self.n_actuators)
        
        # Pad if necessary
        if len(joint_pos) < 8:
            joint_pos = np.pad(joint_pos, (0, 8 - len(joint_pos)))
        if len(joint_vel) < 8:
            joint_vel = np.pad(joint_vel, (0, 8 - len(joint_vel)))
        
        # Fingertip positions
        arm1_fingertip = self._get_site_pos("arm1_touch_site")
        arm2_fingertip = self._get_site_pos("arm2_touch_site")
        
        # Target bar position
        target_bar_pos = self.bar_positions[self.current_target_bar] if len(self.bar_positions) > 0 else np.zeros(3)
        
        obs = np.concatenate([
            base_pos,
            base_quat,
            base_linvel,
            base_angvel,
            joint_pos[:8],
            joint_vel[:8],
            arm1_fingertip,
            arm2_fingertip,
            target_bar_pos,
        ]).astype(np.float32)
        
        return obs
    
    def _get_site_pos(self, site_name: str) -> np.ndarray:
        """Get position of a site."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)
    
    def _get_geom_pos(self, geom_name: str) -> np.ndarray:
        """Get position of a geometry."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            return self.data.geom_xpos[geom_id].copy()
        return np.zeros(3)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random perturbation to initial joint positions
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.05, 0.05, size=self.model.nq)
            noise[:7] = 0  # Don't perturb base pose
            self.data.qpos[:] += noise
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial position
        self.initial_base_pos = self.data.qpos[:3].copy()
        self.current_step = 0
        self.current_target_bar = 0
        
        obs = self._get_obs()
        info = {"initial_pos": self.initial_base_pos.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Scale action from [-1, 1] to joint ranges
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward, reward_info = self._compute_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # Info dict
        info = {
            "step": self.current_step,
            "base_pos": self.data.qpos[:3].copy(),
            "forward_distance": self.data.qpos[0] - self.initial_base_pos[0],
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reward for current state."""
        reward = 0.0
        info = {}
        
        # Current base position
        base_pos = self.data.qpos[:3]
        
        # 1. Forward progress reward
        forward_progress = base_pos[0] - self.initial_base_pos[0]
        forward_reward = forward_progress * 1.0
        reward += forward_reward
        info["forward_reward"] = forward_reward
        
        # 2. Height maintenance (penalize falling)
        height = base_pos[2]
        target_height = 0.5
        height_penalty = -2.0 * max(0, target_height - height)
        reward += height_penalty
        info["height_penalty"] = height_penalty
        
        # 3. Energy efficiency (penalize large actions)
        action_cost = -0.01 * np.sum(np.square(self.data.ctrl))
        reward += action_cost
        info["action_cost"] = action_cost
        
        # 4. Velocity reward (encourage movement)
        velocity_reward = 0.1 * self.data.qvel[0]  # Forward velocity
        reward += velocity_reward
        info["velocity_reward"] = velocity_reward
        
        # 5. Distance to target bar (proximity reward)
        if len(self.bar_positions) > 0:
            target_bar = self.bar_positions[self.current_target_bar]
            arm1_tip = self._get_site_pos("arm1_touch_site")
            arm2_tip = self._get_site_pos("arm2_touch_site")
            
            dist1 = np.linalg.norm(arm1_tip - target_bar)
            dist2 = np.linalg.norm(arm2_tip - target_bar)
            min_dist = min(dist1, dist2)
            
            proximity_reward = 0.5 * np.exp(-2.0 * min_dist)
            reward += proximity_reward
            info["proximity_reward"] = proximity_reward
            
            # Check if we reached the bar
            if min_dist < 0.05:
                reward += 5.0  # Bonus for reaching bar
                info["bar_reached"] = True
                self.current_target_bar = min(self.current_target_bar + 1, len(self.bar_positions) - 1)
        
        return reward, info
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        base_pos = self.data.qpos[:3]
        
        # Terminate if robot falls too low
        if base_pos[2] < 0.1:
            return True
        
        # Terminate if robot tilts too much
        base_quat = self.data.qpos[3:7]
        # Check if upside down (simple check via z-component of up vector)
        up_vec = self._quat_to_up_vec(base_quat)
        if up_vec[2] < 0.0:
            return True
        
        return False
    
    def _quat_to_up_vec(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to up vector."""
        # Simplified rotation of [0, 0, 1] by quaternion
        w, x, y, z = quat
        return np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer = None


# Register the environment with Gymnasium
def register_env():
    """Register the environment with Gymnasium."""
    gym.register(
        id="Brachiation-v0",
        entry_point="src.envs.brachiation_env:BrachiationEnv",
        max_episode_steps=1000,
    )


if __name__ == "__main__":
    # Quick test
    env = BrachiationEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
