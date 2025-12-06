"""
Gymnasium-compatible environment for dual-arm robot traversing walls using MuJoCo.

This environment implements a dual-arm robot that learns to climb over
a series of walls to reach a target on the other side.
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
    Dual-arm wall traversal environment.
    
    The robot must learn to climb over 10 walls (30cm tall, 2cm thick, 15cm apart)
    to reach a target on the other side.
    
    Observation Space:
        - Base position and orientation (7D: xyz + quaternion)
        - Base linear and angular velocity (6D)
        - Joint positions (8D: 5 for arm1, 3 for arm2)
        - Joint velocities (8D)
        - Fingertip positions (6D: 3 for each arm)
        - Target position (3D)
        - Distance to next wall (1D)
        - Walls cleared count normalized (1D)
        Total: 40D
    
    Action Space:
        - Joint position targets for all actuators (8D)
        Continuous, normalized to [-1, 1]
    
    Rewards:
        - Forward progress reward
        - Height maintenance for climbing
        - Energy efficiency penalty
        - Wall clearance bonus
        - Target reach bonus
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        control_freq: int = 50,
        initial_keyframe: Optional[str] = "hanging",
    ):
        """
        Initialize the brachiation environment.
        
        Args:
            render_mode: "human" for interactive viewer, "rgb_array" for image output
            max_episode_steps: Maximum steps per episode
            control_freq: Control frequency in Hz
            initial_keyframe: Name of the keyframe to load at reset, or None to skip
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        self.initial_keyframe = initial_keyframe
        
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
        
        # Wall obstacle course setup
        # 10 walls at x = 0.15, 0.30, 0.45, ..., 1.50 (15cm apart)
        self.wall_positions = np.array([0.15 + 0.15 * i for i in range(10)])
        self.wall_height = 0.30  # 30cm tall
        self.target_pos = np.array([1.7, 0.0, 0.05])  # Target after all walls
        self.walls_cleared = 0
        
    def _get_joint_ranges(self) -> np.ndarray:
        """Get joint position limits."""
        ranges = []
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                ranges.append(self.model.jnt_range[i])
            else:
                ranges.append(np.array([-np.pi, np.pi]))
        return np.array(ranges)
    
    def _get_obs_dim(self) -> int:
        """Calculate observation dimension."""
        # Base pose: 3 (pos) + 4 (quat) = 7
        # Base velocity: 3 (linear) + 3 (angular) = 6
        # Joint positions: 8 (actuated joints)
        # Joint velocities: 8
        # Fingertip positions: 6 (2 arms * 3)
        # Target position: 3
        # Distance to next wall: 1
        # Walls cleared (normalized): 1
        return 7 + 6 + 8 + 8 + 6 + 3 + 1 + 1
    
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
        arm1_fingertip = self._get_site_pos("arm1_tip")
        arm2_fingertip = self._get_site_pos("arm2_tip")
        
        # Distance to next wall (or target if all walls cleared)
        robot_x = base_pos[0]
        next_wall_dist = 0.0
        if self.walls_cleared < len(self.wall_positions):
            next_wall_x = self.wall_positions[self.walls_cleared]
            next_wall_dist = next_wall_x - robot_x
        else:
            next_wall_dist = self.target_pos[0] - robot_x
        
        # Walls cleared normalized (0 to 1)
        walls_cleared_norm = self.walls_cleared / len(self.wall_positions)
        
        obs = np.concatenate([
            base_pos,
            base_quat,
            base_linvel,
            base_angvel,
            joint_pos[:8],
            joint_vel[:8],
            arm1_fingertip,
            arm2_fingertip,
            self.target_pos,
            [next_wall_dist],
            [walls_cleared_norm],
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
        
        # Load the "hanging" keyframe if it exists
        if self.initial_keyframe is not None:
            key_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_KEY,
                self.initial_keyframe,
            )
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
        # Ensure gripper is closed tight for grip
        # ctrl[4] is arm1_gripper_act, set to -1.0 for max grip
        self.data.ctrl[4] = -1.0
        
        # Add small random perturbation to joint positions only (not base or gripper)
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
            noise[:7] = 0  # Don't perturb base pose
            noise[11] = 0  # Don't perturb gripper (arm1_gripper is joint 5, qpos index ~11)
            self.data.qpos[:] += noise
        
        # Step a few times to let gripper settle onto bar
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial position
        self.initial_base_pos = self.data.qpos[:3].copy()
        self.current_step = 0
        self.walls_cleared = 0
        
        obs = self._get_obs()
        info = {"initial_pos": self.initial_base_pos.copy(), "walls_cleared": 0}
        
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
            "walls_cleared": self.walls_cleared,
            **reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reward for current state."""
        reward = 0.0
        info = {}
        
        # Current base position
        base_pos = self.data.qpos[:3]
        robot_x = base_pos[0]
        robot_z = base_pos[2]
        base_quat = self.data.qpos[3:7]
        
        # Update walls cleared count
        old_walls_cleared = self.walls_cleared
        while (self.walls_cleared < len(self.wall_positions) and 
               robot_x > self.wall_positions[self.walls_cleared] + 0.02):  # Past the wall
            self.walls_cleared += 1
        
        # 1. Forward progress reward (normalized by total distance)
        total_distance = self.target_pos[0] - self.initial_base_pos[0]
        forward_progress = (robot_x - self.initial_base_pos[0]) / total_distance
        forward_reward = forward_progress * 2.0
        reward += forward_reward
        info["forward_reward"] = forward_reward
        
        # 2. Wall clearance bonus (big reward for each wall cleared)
        walls_just_cleared = self.walls_cleared - old_walls_cleared
        wall_bonus = walls_just_cleared * 10.0
        reward += wall_bonus
        info["wall_bonus"] = wall_bonus
        
        # 3. Height reward when near walls (encourage climbing)
        if self.walls_cleared < len(self.wall_positions):
            next_wall_x = self.wall_positions[self.walls_cleared]
            dist_to_wall = abs(robot_x - next_wall_x)
            if dist_to_wall < 0.1:  # Close to wall
                # Reward height when near wall (need to be above 30cm to clear)
                height_reward = 2.0 * max(0, robot_z - 0.3) if robot_z > 0.2 else 0
                reward += height_reward
                info["height_reward"] = height_reward
            else:
                info["height_reward"] = 0.0
        else:
            info["height_reward"] = 0.0
        
        # 4. Height maintenance (don't fall, but don't need to stay too high)
        min_height = 0.05
        height_penalty = -5.0 * max(0, min_height - robot_z)
        reward += height_penalty
        info["height_penalty"] = height_penalty
        
        # 5. Energy efficiency (penalize large actions)
        action_cost = -0.005 * np.sum(np.square(self.data.ctrl))
        reward += action_cost
        info["action_cost"] = action_cost
        
        # 6. Velocity reward (encourage forward movement)
        velocity_reward = 0.2 * max(0, self.data.qvel[0])  # Forward velocity only
        reward += velocity_reward
        info["velocity_reward"] = velocity_reward

        # 8. Upright orientation bonus
        up_vec = self._quat_to_up_vec(base_quat)
        upright_bonus = max(0.0, up_vec[2]) * 2.0
        reward += upright_bonus
        info["upright_bonus"] = upright_bonus

        # 9. Angular velocity penalty (encourage stability)
        ang_vel_penalty = -0.1 * np.linalg.norm(self.data.qvel[3:6])
        reward += ang_vel_penalty
        info["ang_vel_penalty"] = ang_vel_penalty
        
        # 7. Target proximity reward (when all walls cleared)
        if self.walls_cleared >= len(self.wall_positions):
            dist_to_target = np.linalg.norm(base_pos[:2] - self.target_pos[:2])
            proximity_reward = 5.0 * np.exp(-2.0 * dist_to_target)
            reward += proximity_reward
            info["proximity_reward"] = proximity_reward
            
            # Huge bonus for reaching target
            if dist_to_target < 0.1:
                reward += 100.0
                info["target_reached"] = True
        else:
            info["proximity_reward"] = 0.0
        
        return reward, info
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        base_pos = self.data.qpos[:3]
        
        # Terminate if robot falls too low
        if base_pos[2] < 0.02:
            return True
        
        # Terminate if robot goes too far backwards
        if base_pos[0] < self.initial_base_pos[0] - 0.3:
            return True
        
        # Terminate if robot tilts too much (upside down)
        base_quat = self.data.qpos[3:7]
        up_vec = self._quat_to_up_vec(base_quat)
        if up_vec[2] < -0.5:  # Significantly inverted
            return True
        
        # Success termination - reached target
        dist_to_target = np.linalg.norm(base_pos[:2] - self.target_pos[:2])
        if dist_to_target < 0.1 and self.walls_cleared >= len(self.wall_positions):
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
