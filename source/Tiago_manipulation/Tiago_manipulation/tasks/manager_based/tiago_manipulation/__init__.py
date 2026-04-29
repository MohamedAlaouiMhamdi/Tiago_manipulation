# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Tiago-Manipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_manipulation_env_cfg:TiagoManipulationEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Pick-and-lift task (Phase 1: fixed-base tabletop cube lift).
gym.register(
    id="Template-Tiago-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_lift_env_cfg:TiagoLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tiago-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_lift_env_cfg:TiagoLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Vision-only pick-and-lift task (policy sees only RGB).
gym.register(
    id="Template-Tiago-Lift-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_lift_vision_env_cfg:TiagoLiftVisionEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tiago-Lift-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_lift_vision_env_cfg:TiagoLiftVisionEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)