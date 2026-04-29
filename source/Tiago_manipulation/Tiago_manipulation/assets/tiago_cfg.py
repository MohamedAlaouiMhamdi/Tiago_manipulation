"""ArticulationCfg for TIAGo (omni base + 7-DoF arm + pal-gripper).

Phase 1 convention: the mobile base is fixed via `fix_root_link` so the arm can
be trained alone on a tabletop pick/lift task. Wheels and casters still exist
as joints but get minimal drive.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_ASSETS_DIR = os.path.dirname(__file__)
TIAGO_USD_PATH = os.path.join(_ASSETS_DIR, "usd_final", "tiago.usd")


TIAGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TIAGO_USD_PATH,
        activate_contact_sensors=True,  # needed for undesired-contact reward
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,   # off again — was destabilizing arm sweeps
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # Phase 1: train arm with a fixed mobile base
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # torso_lift_joint is locked in the USD via tight joint limits at 0.20 m
            "torso_lift_joint": 0.20,
            # arm "home" pose — pointing forward/down over a workspace
            "arm_1_joint": 0.20,
            "arm_2_joint": -1.34,
            "arm_3_joint": -0.20,
            "arm_4_joint": 1.94,
            "arm_5_joint": -1.57,
            "arm_6_joint": 1.37,
            "arm_7_joint": 0.00,
            # gripper: fingers fully open
            "gripper_left_finger_joint": 0.045,
            "gripper_right_finger_joint": 0.045,
            # head: pan neutral, tilt fully downward so the camera looks at the table
            "head_1_joint": 0.0,
            "head_2_joint": -0.7,   # ~-40°
            # wheels at rest
            "wheel_.*_joint": 0.0,
        },
    ),
    actuators={
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_lift_joint"],
            effort_limit_sim=10000.0,
            stiffness=1e6,  # USD already pins via joint limits; drive just matches it
            damping=1e4,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_[1-7]_joint"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_.*_finger_joint"],
            effort_limit_sim=200.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_[1-2]_joint"],
            effort_limit_sim=200.0,
            stiffness=2000.0,   # hold tilted-down head against gravity
            damping=200.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            effort_limit_sim=10.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for TIAGo with fixed base (Phase 1)."""
