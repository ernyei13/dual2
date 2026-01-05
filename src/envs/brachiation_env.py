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
        max_episode_steps: int = 2000,
        control_freq: int = 50,
        initial_keyframe: Optional[str] = "hanging",
        task_mode: str = "traversal",
        curriculum_level: int = 8,  # Start wall (0-9). Higher = easier (closer to goal)
    ):
        """
        Initialize the brachiation environment.
        
        Args:
            render_mode: "human" for interactive viewer, "rgb_array" for image output
            max_episode_steps: Maximum steps per episode
            control_freq: Control frequency in Hz
            initial_keyframe: Name of the keyframe to load at reset, or None to skip
            task_mode: Task to train ("traversal" or "grasping")
            curriculum_level: Starting wall index (0-9). Higher = closer to goal = easier.
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.task_mode = task_mode
        self.control_freq = control_freq
        self.initial_keyframe = initial_keyframe
        self.curriculum_level = curriculum_level
        
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
        self.previous_action = None
        self.prev_distance_to_goal = None  # For distance-based reward
        
        # Original masses for domain randomization
        self.original_masses = self.model.body_mass.copy()

        
        # Viewer for rendering
        self.viewer = None
        self.renderer = None
        
        # Wall obstacle course setup
        # 10 walls at x = 0.15, 0.30, 0.45, ..., 1.50 (15cm apart)
        self.wall_positions = np.array([0.15 + 0.15 * i for i in range(10)])
        self.wall_height = 0.31  # Height of horizontal bars above ground
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
        # Walls cleared normalized (1D)
        # Hand to target wall vector (3D) -- NEW for Reaching Reward
        return 7 + 6 + 8 + 8 + 6 + 3 + 1 + 1 + 3

    def _find_target_wall_idx(self) -> int:
        """Find the index of the wall we should be targeting."""
        # If we are in grasping mode, we spawned near a random wall.
        # We need to know WHICH wall that is.
        # We can find the closest wall in front of us.
        robot_x = self.data.qpos[0]
        
        # Simple heuristic: find nearest wall
        # Only check walls within reasonable range
        dists = np.abs(self.wall_positions - robot_x)
        nearest_idx = np.argmin(dists)
        return nearest_idx
    
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
        
        # Hand to Wall Vector (for Reaching Reward)
        # Find target wall
        target_wall_idx = self._find_target_wall_idx()
        target_wall_x = self.wall_positions[target_wall_idx]
        target_wall_height = self.wall_height # 0.3m is top? Or center? 
        # Wall is cuboid size=(0.01, 0.15, 0.15). Pos is center.
        # Top of wall is pos_z + size_z = 0.15 + 0.15 = 0.3.
        # We want to grab the top.
        target_point = np.array([target_wall_x, 0.0, 0.3])
        
        # Find closest hand (arm1 or arm2)
        dist1 = np.linalg.norm(target_point - arm1_fingertip)
        dist2 = np.linalg.norm(target_point - arm2_fingertip)
        
        if dist1 < dist2:
            hand_to_wall = target_point - arm1_fingertip
        else:
            hand_to_wall = target_point - arm2_fingertip
            
        self.current_hand_to_wall_dist = min(dist1, dist2) # Store for reward
        
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
            hand_to_wall, # NEW
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
    
    def _get_touch_sensor(self, sensor_name: str) -> float:
        """Get value of a touch sensor."""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id >= 0:
            return self.data.sensordata[sensor_id]
        return 0.0

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
            noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
            # Apply noise to base position (x, y, z)
            # Keep orientation (quaternion) stable or apply very small noise if desired
            noise[3:7] = 0.0  # Keep orientation exact for stability
            
            # Don't perturb gripper (arm1_gripper is joint 5, qpos index ~11)
            # Adjust index based on actual model structure if needed, assumig index 11 is correct from previous code
            noise[11] = 0 
            
            self.data.qpos[:] += noise

        # DOMAIN RANDOMIZATION: Randomize link masses (robustness)
        if self.np_random is not None:
            # Vary mass by +/- 20%
            random_mass_scale = self.np_random.uniform(0.8, 1.2, size=self.model.nbody)
            self.model.body_mass[:] = self.original_masses * random_mass_scale

        # Initialize previous action for smoothness calc
        self.previous_action = np.zeros(self.n_actuators)


        # CURRICULUM: Shift starting position
        if self.task_mode == "grasping":
             # Spawn close to a random wall (0-10cm away) -> Changed to 0-2cm for guaranteed reflex demo
             # Pick a random wall index
             wall_idx = self.np_random.integers(0, 10)
             wall_x = self.wall_positions[wall_idx]
             
             # Shift robot to be near this wall. 
             # Keyframe "hanging" approx at 0.2m? 
             # Let's assume keyframe is suitable for first wall (0.15).
             # Shift = wall_x - 0.15 + random_offset
             
             random_offset_x = self.np_random.uniform(-0.02, 0.0) # Very close spawn!
             shift = (wall_x - 0.15) + random_offset_x
             self.data.qpos[0] += shift
             
             # Randomize vertical position slightly
             self.data.qpos[2] += self.np_random.uniform(-0.05, 0.05)

        else:
            # TRAVERSAL Curriculum
            # Use curriculum_level to determine starting wall.
            # Level 8 = Wall 8 (close to goal, easy)
            # Level 0 = Wall 0 (far from goal, hard)
            start_wall_idx = self.curriculum_level
            start_wall_idx = max(0, min(9, start_wall_idx))  # Clamp to valid range
            start_wall_x = self.wall_positions[start_wall_idx]
            
            # Keyframe has robot near 0.15 (wall 0). Shift = target_x - 0.15
            start_x_shift = start_wall_x - 0.15
            self.data.qpos[0] += start_x_shift
        
        # Step a few times to let gripper settle onto bar
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial position
        self.initial_base_pos = self.data.qpos[:3].copy()
        self.current_step = 0
        
        # Initialize distance tracking for reward
        self.prev_distance_to_goal = self.target_pos[0] - self.data.qpos[0]
        
        # Update walls_cleared based on starting position
        self.walls_cleared = 0
        for w_pos in self.wall_positions:
            if self.initial_base_pos[0] > w_pos + 0.02:
                self.walls_cleared += 1
        
        obs = self._get_obs()
        info = {"initial_pos": self.initial_base_pos.copy(), "walls_cleared": 0}
        
        return obs, info
    
    def _apply_grasp_reflex(self, action: np.ndarray) -> np.ndarray:
        """
        Bio-Inspired Grasp Reflex:
        If hand is close to wall or touching it, automatically close the gripper.
        Overrides specific indices in the action array.
        """
        # Distances are computed in _get_obs -> stored in self.current_hand_to_wall_dist
        # But we need specific arm distances here.
        # Let's re-query or compute simply.
        
        # Thresholds
        GRASP_DIST_THRESHOLD = 0.05 # 5cm proximity triggers reflex
        
        # Arm 1
        touch1 = self._get_touch_sensor("arm1_touch")
        # We need hand-to-wall dist for arm 1 specifically. 
        # For efficiency, let's just check touch first as it's most robust.
        
        # Override if touching
        # arm1_gripper is idx 4 (actuator index) based on xml
        # arm2_gripper is idx 7
        
        # Note: action is scaled [-1, 1]. -1 is closed (based on reset).
        
        if touch1 > 0.01:
            action[4] = -1.0 # Force close
            
        # Arm 2
        touch2 = self._get_touch_sensor("arm2_touch")
        if touch2 > 0.01:
            action[7] = -1.0 # Force close
            
        # Proximity Reflex (Visual/Distance based)
        # If we have the distance data available (we do in _get_obs but relying on internal state is tricky order-wise)
        # Let's trust the touch sensor primarily for the "Reflex".
        # Pure proximity reflex might be annoying if we want to release?
        # A true reflex usually inhibits release if holding something.
        # Let's stick to Contact Reflex for now. It's cleaner.
        
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Scale action from [-1, 1] to joint ranges
        action = np.clip(action, -1.0, 1.0)
        
        # Apply Bio-Inspired Reflex (Override actions)
        action = self._apply_grasp_reflex(action)
        
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Calculate smoothness penalty (before updating previous_action)
        # Penalize large changes in action
        if self.previous_action is not None:
            action_diff = action - self.previous_action
            self.smoothness_penalty = -0.1 * np.mean(np.square(action_diff))
        else:
            self.smoothness_penalty = 0.0
            
        self.previous_action = action.copy()
        
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
        """
        MULTI-COMPONENT REWARD FOR BRACHIATION
        
        Combines:
        1. Distance progress toward goal
        2. Height maintenance (stay above ground)
        3. Bar reaching reward (get hand close to bars)
        4. Grip reward (holding onto bars)
        5. Action smoothness
        """
        info = {}
        
        # === DISTANCE PROGRESS REWARD ===
        robot_x = self.data.qpos[0]
        current_distance = self.target_pos[0] - robot_x
        
        # Progress = how much closer we got (positive = good)
        progress = self.prev_distance_to_goal - current_distance
        forward_reward = progress * 20.0  # Scale for visibility
        info["forward_progress"] = progress
        info["current_distance"] = current_distance
        
        # Update for next step
        self.prev_distance_to_goal = current_distance
        
        # === HEIGHT MAINTENANCE REWARD ===
        robot_z = self.data.qpos[2]
        # Optimal height is around bar level (~0.31m)
        target_height = 0.31
        height_error = abs(robot_z - target_height)
        height_reward = -2.0 * height_error  # Penalize being too high or low
        info["height_reward"] = height_reward
        
        # === BAR REACHING REWARD ===
        # Encourage hands to approach the next bar
        reaching_reward = 0.0
        if hasattr(self, 'current_hand_to_wall_dist'):
            # Reward for getting hand close to bar (inverse distance)
            reaching_reward = max(0, 0.1 - self.current_hand_to_wall_dist) * 5.0
        info["reaching_reward"] = reaching_reward
        
        # === GRIP REWARD ===
        # Reward for touching/holding bars
        touch1 = self._get_touch_sensor("arm1_touch")
        touch2 = self._get_touch_sensor("arm2_touch")
        grip_reward = (min(touch1, 1.0) + min(touch2, 1.0)) * 0.5
        info["grip_reward"] = grip_reward
        
        # === ACTION SMOOTHNESS ===
        smoothness_reward = self.smoothness_penalty if hasattr(self, 'smoothness_penalty') else 0.0
        info["smoothness_reward"] = smoothness_reward
        
        # === SMALL ACTION COST (prevent wild flailing) ===
        action_cost = -0.005 * np.mean(np.square(self.data.ctrl))
        info["action_cost"] = action_cost
        
        # === TRACK WALLS CLEARED (for info only) ===
        while (self.walls_cleared < len(self.wall_positions) and 
               robot_x > self.wall_positions[self.walls_cleared] + 0.02):
            self.walls_cleared += 1
        info["walls_cleared"] = self.walls_cleared
        
        # === SUCCESS BONUS (reach target) ===
        dist_to_target = np.linalg.norm(self.data.qpos[:2] - self.target_pos[:2])
        if dist_to_target < 0.15 and self.walls_cleared >= len(self.wall_positions):
            success_bonus = 100.0
            info["target_reached"] = True
        else:
            success_bonus = 0.0
        
        # === TOTAL REWARD ===
        reward = forward_reward + height_reward + reaching_reward + grip_reward + smoothness_reward + action_cost + success_bonus
        info["total_reward"] = reward
        
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
                from mujoco import viewer as mj_viewer
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
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
        max_episode_steps=2000,
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
