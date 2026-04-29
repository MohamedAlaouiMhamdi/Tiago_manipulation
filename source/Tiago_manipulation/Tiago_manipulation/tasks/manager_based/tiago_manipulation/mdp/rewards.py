# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def _task_success_bool(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_z_range: tuple[float, float],
    xy_radius: float,
    gripper_open_threshold: float,
    max_velocity: float,
    object_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    obj_pos = obj.data.root_pos_w
    dx = obj_pos[:, 0] - target_xy[0]
    dy = obj_pos[:, 1] - target_xy[1]
    xy_ok = (dx * dx + dy * dy) < (xy_radius * xy_radius)
    z_ok = (obj_pos[:, 2] > target_z_range[0]) & (obj_pos[:, 2] < target_z_range[1])

    finger_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    open_ok = (finger_pos > gripper_open_threshold).all(dim=-1)

    obj_vel = obj.data.root_lin_vel_w
    rest_ok = obj_vel.norm(dim=-1) < max_velocity

    return xy_ok & z_ok & open_ok & rest_ok


def task_success(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_z_range: tuple[float, float],
    xy_radius: float,
    gripper_open_threshold: float,
    max_velocity: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["gripper_.*_finger_joint"]
    ),
) -> torch.Tensor:
    """Termination signal — returns a BOOL tensor (cube placed + open + at rest)."""
    return _task_success_bool(
        env, target_xy, target_z_range, xy_radius,
        gripper_open_threshold, max_velocity, object_cfg, gripper_cfg,
    )


def task_success_reward(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_z_range: tuple[float, float],
    xy_radius: float,
    gripper_open_threshold: float,
    max_velocity: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["gripper_.*_finger_joint"]
    ),
) -> torch.Tensor:
    """Reward variant — same predicate, returns a FLOAT tensor."""
    return _task_success_bool(
        env, target_xy, target_z_range, xy_radius,
        gripper_open_threshold, max_velocity, object_cfg, gripper_cfg,
    ).float()


def object_goal_distance_dead_zone(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    dead_zone_radius: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Same shape as the lift task's `object_goal_distance` but ZEROED within
    `dead_zone_radius` of the goal. Habitat-style: once the cube is "at goal",
    no more shaping reward — hovering stops paying."""
    from isaaclab.utils.math import combine_frame_transforms

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)

    above_thresh = obj.data.root_pos_w[:, 2] > minimal_height
    outside_dead_zone = distance > dead_zone_radius
    gate = (above_thresh & outside_dead_zone).float()
    return gate * (1.0 - torch.tanh(distance / std))


def slack(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 reward every step. Use with a small negative weight (e.g. −0.005)
    for time-pressure bleed (Habitat-style)."""
    return torch.ones(env.num_envs, device=env.device)


def closed_gripper_near_goal(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    xy_radius: float,
    closed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["gripper_.*_finger_joint"]
    ),
) -> torch.Tensor:
    """Binary: 1.0 when cube is near the bin AND the gripper is COMMANDED closed.

    Used with a NEGATIVE weight to punish "hovering with cube held" at the goal
    state. Pickup elsewhere on the table is unaffected because the penalty only
    fires when xy is within `xy_radius` of the bin center.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    obj_pos = obj.data.root_pos_w
    dx = obj_pos[:, 0] - target_xy[0]
    dy = obj_pos[:, 1] - target_xy[1]
    xy_ok = (dx * dx + dy * dy) < (xy_radius * xy_radius)

    joint_ids = gripper_cfg.joint_ids
    if joint_ids is None or (hasattr(joint_ids, "__len__") and len(joint_ids) == 0):
        joint_ids, _ = robot.find_joints(gripper_cfg.joint_names)

    finger_target = robot.data.joint_pos_target[:, joint_ids]
    closed_ok = (finger_target < closed_threshold).all(dim=-1)

    return (xy_ok & closed_ok).float()


def time_holding_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Per-step penalty proportional to elapsed episode time, fires while the cube is
    held in the air. Forces the policy to release before the episode ends — without
    this, hovering pays a constant ~58/step which crowds out any release exploration.

    Returns episode_time_seconds when cube z > height_threshold, else 0. With a
    negative weight the longer the cube stays airborne the more painful it gets.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    in_air = obj.data.root_pos_w[:, 2] > height_threshold
    t_seconds = env.episode_length_buf.float() * env.step_dt
    return in_air.float() * t_seconds


def placement_progress(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_z: float,
    xy_sigma: float = 0.10,
    z_sigma: float = 0.05,
    finger_full_open: float = 0.045,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["gripper_.*_finger_joint"]
    ),
) -> torch.Tensor:
    """Smooth multiplicative reward for HOW close the placement is.

    Returns a value in [0, 1] equal to xy_score * z_score * open_score, where each
    factor is a smooth Gaussian/clamp around the target. Provides PPO with a
    continuous gradient instead of an all-or-nothing cliff like `object_placed`.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    obj_pos = obj.data.root_pos_w
    dx = obj_pos[:, 0] - target_xy[0]
    dy = obj_pos[:, 1] - target_xy[1]
    xy_score = torch.exp(-(dx * dx + dy * dy) / (2 * xy_sigma * xy_sigma))
    z_score = torch.exp(-(obj_pos[:, 2] - target_z) ** 2 / (2 * z_sigma * z_sigma))

    joint_ids = gripper_cfg.joint_ids
    if joint_ids is None or (hasattr(joint_ids, "__len__") and len(joint_ids) == 0):
        joint_ids, _ = robot.find_joints(gripper_cfg.joint_names)

    finger_target = robot.data.joint_pos_target[:, joint_ids]
    finger_actual = robot.data.joint_pos[:, joint_ids]
    open_signal = torch.maximum(finger_target, finger_actual).mean(dim=-1)
    open_score = (open_signal / finger_full_open).clamp(0.0, 1.0)

    return xy_score * z_score * open_score


def gripper_open_near_goal(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    xy_radius: float,
    gripper_open_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["gripper_.*_finger_joint"]
    ),
) -> torch.Tensor:
    """Binary nudge: 1.0 when cube is near the place target AND the policy is asking
    for the gripper to be open.

    Resolves joint indices on the fly so this works whether the SceneEntityCfg was
    pre-resolved by the manager or not. Treats EITHER the commanded target OR the
    actual joint position exceeding the threshold as "open" — covers both the case
    where the cube blocks the fingers (only target is high) and the case where the
    cube has been released (actual position is high).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    obj_pos = obj.data.root_pos_w
    dx = obj_pos[:, 0] - target_xy[0]
    dy = obj_pos[:, 1] - target_xy[1]
    xy_ok = (dx * dx + dy * dy) < (xy_radius * xy_radius)

    # Resolve joint indices from names if the manager didn't.
    joint_ids = gripper_cfg.joint_ids
    if joint_ids is None or (hasattr(joint_ids, "__len__") and len(joint_ids) == 0):
        joint_ids, _ = robot.find_joints(gripper_cfg.joint_names)

    finger_target = robot.data.joint_pos_target[:, joint_ids]
    finger_actual = robot.data.joint_pos[:, joint_ids]
    open_ok = (
        (finger_target > gripper_open_threshold) | (finger_actual > gripper_open_threshold)
    ).all(dim=-1)

    return (xy_ok & open_ok).float()


def object_placed(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_z_range: tuple[float, float],
    xy_radius: float,
    gripper_open_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["gripper_.*_finger_joint"]
    ),
) -> torch.Tensor:
    """Binary reward: 1.0 if the object is at the place target AND the gripper is open.

    - xy inside a cylinder of radius `xy_radius` around `target_xy` (world frame)
    - z inside the open interval `target_z_range` (world frame)
    - BOTH gripper finger joints opened past `gripper_open_threshold`

    Use this to explicitly reward a release — not just hovering the object over the spot.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    obj_pos = obj.data.root_pos_w  # (N, 3)
    dx = obj_pos[:, 0] - target_xy[0]
    dy = obj_pos[:, 1] - target_xy[1]
    xy_ok = (dx * dx + dy * dy) < (xy_radius * xy_radius)
    z_ok = (obj_pos[:, 2] > target_z_range[0]) & (obj_pos[:, 2] < target_z_range[1])

    finger_ids = gripper_cfg.joint_ids
    finger_pos = robot.data.joint_pos[:, finger_ids]   # (N, 2)
    open_ok = (finger_pos > gripper_open_threshold).all(dim=-1)

    return (xy_ok & z_ok & open_ok).float()
