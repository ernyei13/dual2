from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import SceneCfg
from omni.isaac.lab.robots import ArticulationCfg, JointControllerCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab.sim as sim_utils

##
# Robot and Controller configuration
##

SO100_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/urdf/robot.urdf", # Corrected path to match actual file
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
    # For this example, we'll use position control.
    # You might want to switch to effort or velocity control later.
    actuators={
        "all_joints": JointControllerCfg(
            joint_names=[".*"], # Actuate all joints
            stiffness=800.0,
            damping=40.0,
        ),
    }
)

##
# Scene definition
##

@configclass
class BrachiationSceneCfg(SceneCfg):
    """Configuration for the brachiation scene."""
    # Add a ground plane
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    # Add the robot
    robot: ArticulationCfg = SO100_CFG

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

    # Define observations
    observations: ObsGroup = ObsGroup(
        # A group for policy-related observations
        policy=ObsGroup(
            # Example: Joint positions and velocities
            joint_pos=ObsTerm(func="joint_pos_norm"),
            joint_vel=ObsTerm(func="joint_vel_rel"),
        )
    )

    # Define actions
    actions = {
        # Corresponds to the "all_joints" actuator defined in SO100_CFG
        "all_joints": {
            "func": "joint_pos_abs",
            "scale": 1.0,
            "offset": 0.0,
        }
    }

    # Define rewards
    rewards = {
        # TODO: Define your reward functions here
        # Example: Reward for moving forward
        # "forward_progress": RewTerm(func=...),
    }

    # Define terminations
    terminations = {
        # Terminate when the episode timer runs out
        "time_out": DoneTerm(func="time_out", time_out=True),
        # TODO: Add other termination conditions (e.g., robot falls)
    }

    def __post_init__(self):
        """Post-initialization checks."""
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 8.0, 3.0)
        # Set the observation and action spaces based on the robot
        self.observations["policy"].enable_corruption = self.enable_corruption
        self.rewards.enable_corruption = self.enable_corruption
        # Set the default robot command tag
        self.commands.robot_command.name = "all_joints"