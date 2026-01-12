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
        max_episode_steps: int = 10000,
        control_freq: int = 50,
        initial_keyframe: Optional[str] = "hanging",
        task_mode: str = "traversal",
        curriculum_level: int = 0,  # Start at wall 0 (first wall)
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
        
        # Ensure BOTH grippers are closed tight for grip
        self.data.ctrl[4] = -1.0  # arm1_gripper_act
        self.data.ctrl[7] = -0.8  # arm2_gripper_act
        
        # Minimal random perturbation - too much causes instability
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.005, 0.005, size=self.model.nq)
            # Keep base position and orientation stable
            noise[:7] = 0.0  # No noise on free joint
            # Don't perturb grippers
            noise[11] = 0  # arm1_gripper
            noise[14] = 0  # arm2_gripper
            self.data.qpos[:] += noise

        # DOMAIN RANDOMIZATION: Randomize link masses (robustness) - reduced range
        if self.np_random is not None:
            # Vary mass by +/- 10%
            random_mass_scale = self.np_random.uniform(0.9, 1.1, size=self.model.nbody)
            self.model.body_mass[:] = self.original_masses * random_mass_scale

        # Initialize previous action for smoothness calc
        self.previous_action = np.zeros(self.n_actuators)


        # CURRICULUM: Shift starting position
        if self.task_mode == "grasping":
            # GRASPING MODE: Stay at wall 1 (keyframe position)
            # No shift - robot starts at keyframe position gripping bar 1
            pass

        else:
            # TRAVERSAL Curriculum
            # Keyframe positions robot at wall 0 (x=0.15)
            # Use curriculum_level to shift from there
            # Level 0 = stay at keyframe position (wall 0)
            # Level > 0 = move forward (easier, closer to goal)
            keyframe_wall = 0
            start_wall_idx = self.curriculum_level
            start_wall_idx = max(0, min(9, start_wall_idx))
            
            if start_wall_idx != keyframe_wall:
                target_x = self.wall_positions[start_wall_idx]
                keyframe_x = self.wall_positions[keyframe_wall]  # 0.15
                shift = target_x - keyframe_x
                self.data.qpos[0] += shift
        
        # Step a few times to let physics settle (fewer steps to prevent instability)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            # Check for simulation instability
            if np.any(np.isnan(self.data.qpos)) or np.any(np.abs(self.data.qpos) > 100):
                # Reset to keyframe if unstable
                if self.initial_keyframe is not None:
                    key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.initial_keyframe)
                    if key_id >= 0:
                        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
                break
        
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
        info = {"initial_pos": self.initial_base_pos.copy(), "walls_cleared": self.walls_cleared}
        
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
        """Dispatch to appropriate reward function based on task mode."""
        if self.task_mode == "grasping":
            return self._compute_grasping_reward()
        else:
            return self._compute_traversal_reward()
    
    def _compute_grasping_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        GRASPING/BALANCING REWARD
        
        Train the robot to:
        1. Grip the bar firmly
        2. Hold itself up (not fall)
        3. Stay stable (minimal swinging)
        4. Keep body upright
        """
        info = {}
        
        robot_x = self.data.qpos[0]
        robot_z = self.data.qpos[2]
        robot_vx = self.data.qvel[0]
        robot_vz = self.data.qvel[2]
        
        # Get touch sensors
        touch1 = self._get_touch_sensor("arm1_touch")
        touch2 = self._get_touch_sensor("arm2_touch")
        
        # === 1. GRIP REWARD (MAIN) ===
        # Big reward for maintaining grip on the bar
        grip_strength = min(touch1, 1.0) + min(touch2, 1.0)
        grip_reward = 3.0 * grip_strength  # Up to 6.0 per step if both gripping
        info["grip_reward"] = grip_reward
        info["touch1"] = touch1
        info["touch2"] = touch2
        
        # === 2. HEIGHT REWARD ===
        # Reward for staying at proper hanging height (base at z=0.09 for straight hang)
        target_height = 0.09  # Base height when hanging straight below bar
        height_error = abs(robot_z - target_height)
        height_reward = 1.0 * np.exp(-10.0 * height_error**2)
        info["height_reward"] = height_reward
        
        # === 3. STABILITY REWARD ===
        # Penalize excessive movement/swinging (linear velocity)
        velocity_magnitude = np.sqrt(robot_vx**2 + robot_vz**2)
        stability_reward = 0.5 * np.exp(-2.0 * velocity_magnitude)
        info["stability_reward"] = stability_reward
        
        # === 3b. ANGULAR VELOCITY PENALTY ===
        # Mild penalty for rotation to encourage stability (but not too harsh)
        angular_vel = self.data.qvel[3:6]
        angular_speed = np.linalg.norm(angular_vel)
        # Exponential decay penalty - small for low angular speed, caps at -1.0
        angular_penalty = -1.0 * (1.0 - np.exp(-0.1 * angular_speed))
        info["angular_penalty"] = angular_penalty
        
        # === 4. UPRIGHT REWARD ===
        # Reward for keeping body orientation stable - CRITICAL
        base_quat = self.data.qpos[3:7]
        up_vec = self._quat_to_up_vec(base_quat)
        # MUCH stronger reward for staying upright, HUGE penalty for inverting
        if up_vec[2] > 0.5:
            # Upright: bonus for being stable
            upright_reward = 4.0 * up_vec[2]  # Up to 4.0 when perfectly upright
        elif up_vec[2] > 0:
            # Tilted but not inverted: smaller reward
            upright_reward = 2.0 * up_vec[2]
        else:
            # Inverted: MASSIVE penalty to discourage this
            upright_reward = 10.0 * up_vec[2]  # Up to -10.0 when fully inverted
        info["upright_reward"] = upright_reward
        info["up_vec_z"] = up_vec[2]
        
        # === 5. POSITION REWARD ===
        # Reward for staying near the bar (x=0.15)
        bar_x = 0.15
        x_error = abs(robot_x - bar_x)
        position_reward = 0.3 * np.exp(-5.0 * x_error**2)
        info["position_reward"] = position_reward
        
        # === 6. FALL PENALTY ===
        # Strong penalty if falling
        if robot_z < 0.1:
            fall_penalty = -5.0
        elif robot_z < 0.15:
            fall_penalty = -2.0
        else:
            fall_penalty = 0.0
        info["fall_penalty"] = fall_penalty
        
        # === 7. GRIP LOSS PENALTY ===
        # Penalty if not gripping at all
        if grip_strength < 0.1:
            no_grip_penalty = -3.0
        else:
            no_grip_penalty = 0.0
        info["no_grip_penalty"] = no_grip_penalty
        
        # === TOTAL REWARD ===
        reward = (grip_reward + height_reward + stability_reward + angular_penalty +
                  upright_reward + position_reward + fall_penalty + no_grip_penalty)
        info["total_reward"] = reward
        
        return reward, info
    
    def _compute_traversal_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        TRAVERSAL REWARD - for moving across bars
        
        Components:
        1. Forward progress (MAIN)
        2. Velocity bonus
        3. Wall clearance bonus
        4. Height maintenance
        5. Grip reward
        6. Time penalty
        """
        info = {}
        
        robot_x = self.data.qpos[0]
        robot_z = self.data.qpos[2]
        robot_vx = self.data.qvel[0]  # Forward velocity
        
        # Clamp velocity to prevent reward explosion from simulation instability
        robot_vx = np.clip(robot_vx, -5.0, 5.0)
        
        # Get joint velocities for swing bonus
        joint_velocities = self.data.qvel[6:]  # Skip base velocities (6 DOF)
        joint_speed = np.sum(np.abs(joint_velocities))
        
        # === 1. DISTANCE PROGRESS REWARD (DOMINANT) ===
        current_distance = self.target_pos[0] - robot_x
        progress = self.prev_distance_to_goal - current_distance
        # Clamp progress to reasonable range (max ~15cm per step at 50Hz)
        progress = np.clip(progress, -0.1, 0.1)
        # Strong reward for forward progress
        forward_reward = progress * 100.0
        info["forward_progress"] = progress
        info["current_distance"] = current_distance
        self.prev_distance_to_goal = current_distance
        
        # === 2. VELOCITY BONUS (ENCOURAGE FAST MOVEMENT) ===
        # Reward forward velocity - move fast!
        velocity_bonus = 2.0 * max(0, robot_vx)  # Higher weight for speed
        info["velocity_bonus"] = velocity_bonus
        
        # === 2b. SWING BONUS (ENCOURAGE JOINT MOVEMENT) ===
        # Reward for moving joints - don't be lazy!
        swing_bonus = 0.1 * min(joint_speed, 10.0)  # Cap to prevent instability reward
        info["swing_bonus"] = swing_bonus
        
        # === 3. WALL CLEARANCE BONUS (BIG MILESTONE) ===
        prev_walls = self.walls_cleared
        while (self.walls_cleared < len(self.wall_positions) and 
               robot_x > self.wall_positions[self.walls_cleared] + 0.02):
            self.walls_cleared += 1
        
        # HUGE bonus for clearing each wall - this is the real objective
        walls_just_cleared = self.walls_cleared - prev_walls
        wall_bonus = walls_just_cleared * 100.0
        info["wall_bonus"] = wall_bonus
        info["walls_cleared"] = self.walls_cleared
        
        # === 4. HEIGHT MAINTENANCE (SECONDARY) ===
        target_height = 0.31  # Bar level
        height_error = abs(robot_z - target_height)
        # Reward for being at bar height
        height_reward = 0.3 * np.exp(-5.0 * height_error**2)
        # STRONG penalty for falling below bar level
        if robot_z < 0.2:
            height_reward -= 3.0 * (0.2 - robot_z)
        info["height_reward"] = height_reward
        
        # === 5. GRIP REWARD ===
        # Reward gripping, but give extra for gripping + moving
        touch1 = self._get_touch_sensor("arm1_touch")
        touch2 = self._get_touch_sensor("arm2_touch")
        is_gripping = (touch1 > 0.01) or (touch2 > 0.01)
        is_moving = abs(robot_vx) > 0.05  # Moving forward or swinging
        
        if is_gripping and is_moving:
            grip_reward = 0.5  # Good: grip + movement
        elif is_gripping:
            grip_reward = 0.1  # Small reward for just holding on
        else:
            grip_reward = -1.0  # Strong penalty for not gripping (falling)
        info["grip_reward"] = grip_reward
        
        # === 6. ALIVE BONUS (conditional on height) ===
        # Only give alive bonus if robot is at a reasonable height
        if robot_z > 0.15:
            alive_bonus = 0.2  # Small bonus for staying up
        else:
            alive_bonus = 0.0
        info["alive_bonus"] = alive_bonus
        
        # === 7. TIME PENALTY (ANTI-STALLING) ===
        # Constant penalty per step to discourage sitting still
        # But reduced so it doesn't dominate when robot is doing well
        time_penalty = -0.3
        info["time_penalty"] = time_penalty
        
        # === 7. ACTION COST (MINIMAL) ===
        # Very low penalty - we WANT fast aggressive movements!
        action_cost = -0.0001 * np.sum(np.square(self.data.ctrl))
        info["action_cost"] = action_cost
        
        # === 8. SUCCESS BONUS ===
        dist_to_target = np.linalg.norm(self.data.qpos[:2] - self.target_pos[:2])
        if dist_to_target < 0.15 and self.walls_cleared >= len(self.wall_positions):
            success_bonus = 500.0  # HUGE bonus for completing the course
            info["target_reached"] = True
        else:
            success_bonus = 0.0
        
        # === TOTAL REWARD ===
        reward = (forward_reward + velocity_bonus + swing_bonus + wall_bonus + 
                  height_reward + grip_reward + alive_bonus + time_penalty + 
                  action_cost + success_bonus)
        info["total_reward"] = reward
        
        return reward, info
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        base_pos = self.data.qpos[:3]
        
        # Terminate if robot falls too low (very lenient)
        if base_pos[2] < -0.1:  # Allow going below ground slightly
            return True
        
        # For grasping mode, be VERY lenient - let robot learn for 100+ steps
        # Only terminate if robot falls way below ground
        if self.task_mode == "grasping":
            # Only terminate if completely fallen through ground
            if base_pos[2] < -0.5:
                return True
            return False
        
        # Terminate if robot goes too far backwards (more than 1m back from start)
        if base_pos[0] < self.initial_base_pos[0] - 1.0:
            return True
        
        # Terminate if robot tilts completely upside down
        base_quat = self.data.qpos[3:7]
        up_vec = self._quat_to_up_vec(base_quat)
        if up_vec[2] < -0.8:  # Very lenient - almost completely inverted
            return True
        
        # Success termination - reached target (only if made real progress)
        dist_to_target = np.linalg.norm(base_pos[:2] - self.target_pos[:2])
        started_near_goal = self.initial_base_pos[0] > 1.5
        if dist_to_target < 0.1 and self.walls_cleared >= len(self.wall_positions) and not started_near_goal:
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
