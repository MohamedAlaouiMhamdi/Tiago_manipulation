"""Pick-and-lift env for TIAGo (Phase 1 — fixed base + tabletop cube).

Mirrors the Franka Cube-Lift task, swapping:
  - Robot       → TIAGo (fixed-base variant) loaded from our local USD.
  - Arm action  → 7-DoF joint-position targets on ``arm_[1-7]_joint``.
  - Gripper     → binary open/close on ``gripper_*_finger_joint``.
  - EE frame    → ``arm_7_link`` + wrist offset.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG

from Tiago_manipulation.assets.tiago_cfg import TIAGO_CFG

from . import mdp


# Marker for the end-effector FrameTransformer visualizer.
# Declared at module level so InteractiveScene doesn't try to treat it as an asset.
_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)
_EE_MARKER_CFG.prim_path = "/Visuals/EEFrame"

# Table placement in world (robot is at the origin facing +X).
# TABLE_X is the forward distance from TIAGo's base to the table CENTER.
# Push it back if the arm keeps clipping into the near edge.
TABLE_X = 0.95
TABLE_TOP_Z = 0.65

# Table dimensions (meters).
_TABLE_W = 0.90   # depth (x)
_TABLE_D = 1.20   # width (y)
_TABLE_TOP_THICKNESS = 0.04
_TABLE_LEG_SIZE = 0.05
_TABLE_WOOD = (0.45, 0.30, 0.18)
_LEG_HEIGHT = TABLE_TOP_Z - _TABLE_TOP_THICKNESS
_LEG_INSET_X = _TABLE_W / 2 - _TABLE_LEG_SIZE / 2 - 0.02
_LEG_INSET_Y = _TABLE_D / 2 - _TABLE_LEG_SIZE / 2 - 0.02
_LEG_Z = _LEG_HEIGHT / 2


def _leg_cfg(dx: float, dy: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg_" + f"{'p' if dx>0 else 'n'}{'p' if dy>0 else 'n'}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TABLE_X + dx, dy, _LEG_Z]),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIZE, _TABLE_LEG_SIZE, _LEG_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_TABLE_WOOD, roughness=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


##
# Scene
##


@configclass
class TiagoLiftSceneCfg(InteractiveSceneCfg):
    """Tabletop scene: TIAGo (fixed base), table, red block, ground, light."""

    robot = TIAGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Procedural house-style table: 4 legs + top slab, all derived from TABLE_TOP_Z.
    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TABLE_X, 0.0, TABLE_TOP_Z - _TABLE_TOP_THICKNESS / 2]),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_W, _TABLE_D, _TABLE_TOP_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_TABLE_WOOD, roughness=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    table_leg_fl = _leg_cfg(+_LEG_INSET_X, +_LEG_INSET_Y)
    table_leg_fr = _leg_cfg(+_LEG_INSET_X, -_LEG_INSET_Y)
    table_leg_bl = _leg_cfg(-_LEG_INSET_X, +_LEG_INSET_Y)
    table_leg_br = _leg_cfg(-_LEG_INSET_X, -_LEG_INSET_Y)

    # Place-target receptacle: the black sorting bin on the right side of the table.
    target_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetBox",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TABLE_X, +0.30, TABLE_TOP_Z]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/black_sorting_bin.usd"
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[TABLE_X - 0.30, -0.30, TABLE_TOP_Z + 0.03],   # near the FRONT edge of the table, robot's left side
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # ContactSensor: watches the arm links so we can penalize arm-table / arm-bin bumps.
    # Gripper finger links are EXCLUDED — they MUST contact the cube during a grasp.
    arm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_[1-6]_link",
        update_period=0.0,
        history_length=1,
    )

    # FrameTransformer: track the gripper grasp point (Xform inside the arm_7_link subtree).
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=_EE_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_7_link",
                name="end_effector",
                # rough offset from arm_7_link frame to the finger-tip grasp point (meters)
                offset=OffsetCfg(pos=[0.0, 0.0, 0.20]),
            ),
        ],
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    # Goal = ON the bin surface, not hovering above. Reaching the goal now means
    # the cube is physically placed, not held in mid-air.
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm_7_link",
        resampling_time_range=(100.0, 100.0),  # > episode_length_s so goal stays fixed per episode
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(TABLE_X, TABLE_X),                 # fixed across episodes
            pos_y=(+0.30, +0.30),
            pos_z=(TABLE_TOP_Z + 0.04, TABLE_TOP_Z + 0.04),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_[1-7]_joint"],
        scale=1.0,                  # was 0.5 — too narrow for the 60 cm cube→bin sweep
        use_default_offset=True,
    )
    # Continuous gripper: policy outputs a target finger position (clamped to limits).
    # Gaussian exploration naturally samples the full [0, 0.045] range, so the
    # policy can learn to OPEN via gradients rather than rare exploration luck.
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_.*_finger_joint"],
        scale=0.045,                # network output ∈ [-1, 1] → ±45 mm
        use_default_offset=True,    # default = init joint pos = 0.045 (open)
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # FIXED object spawn — every episode starts the block at exactly its init_state
    # position. Once the policy converges, widen these ranges to add variation.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.3},   # 30 cm tanh width — gradient exists from TIAGo's home pose
        weight=3.0,            # amplified so reach signal dominates smoothness penalties
    )
    # `minimal_height` = 5 cm above table → a GATE for "cube is actually lifted".
    # When placed on the bin the cube goes BELOW this gate and the lift/track
    # rewards switch off, so the policy stops being paid for mid-air hovering.
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": TABLE_TOP_Z + 0.05},
        weight=15.0,         # restore — fast carry learning
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.5, "minimal_height": TABLE_TOP_Z + 0.05, "command_name": "object_pose"},
        weight=30.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": TABLE_TOP_Z + 0.05, "command_name": "object_pose"},
        weight=10.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_[1-7]_joint"])},
    )
    arm_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,          # 10× stronger so it actually changes behavior
        params={"sensor_cfg": SceneEntityCfg("arm_contacts"), "threshold": 1.0},
    )
    # Nudge: rewards merely OPENING the gripper while the cube is near the bin —
    # gives PPO a gradient for the discrete "release" action, breaking the chicken-and-egg
    # of "policy never tries opening because closing always paid".
    open_near_goal = RewTerm(
        func=mdp.gripper_open_near_goal,
        weight=60.0,
        params={
            "target_xy": (TABLE_X, 0.30),
            "xy_radius": 0.20,
            "gripper_open_threshold": 0.030,
        },
    )
    # Big one-shot bonus when full success is reached (cube placed, gripper open, at rest).
    task_success_bonus = RewTerm(
        func=mdp.task_success_reward,
        weight=500.0,
        params={
            "target_xy": (TABLE_X, 0.30),
            "target_z_range": (TABLE_TOP_Z, TABLE_TOP_Z + 0.15),
            "xy_radius": 0.15,
            "gripper_open_threshold": 0.030,
            "max_velocity": 0.10,
        },
    )
    object_placed = RewTerm(
        func=mdp.object_placed,
        weight=80.0,
        params={
            "target_xy": (TABLE_X, 0.30),
            "target_z_range": (TABLE_TOP_Z, TABLE_TOP_Z + 0.15),
            "xy_radius": 0.15,
            "gripper_open_threshold": 0.035,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Object fell off the table — triggers when z drops >15cm below the table surface.
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_TOP_Z - 0.15, "asset_cfg": SceneEntityCfg("object")},
    )
    # Success! Cube placed, gripper open, cube at rest → end episode and reset to home.
    # This avoids needing the policy to learn "return home" as a separate skill.
    task_done = DoneTerm(
        func=mdp.task_success,
        params={
            "target_xy": (TABLE_X, 0.30),
            "target_z_range": (TABLE_TOP_Z, TABLE_TOP_Z + 0.15),
            "xy_radius": 0.15,
            "gripper_open_threshold": 0.030,
            "max_velocity": 0.10,
        },
    )


##
# Env cfg
##


@configclass
class TiagoLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: TiagoLiftSceneCfg = TiagoLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0   # pick-and-place needs more time than pure lift
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024   # headroom for 4096 envs + self-collisions
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class TiagoLiftEnvCfg_PLAY(TiagoLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
