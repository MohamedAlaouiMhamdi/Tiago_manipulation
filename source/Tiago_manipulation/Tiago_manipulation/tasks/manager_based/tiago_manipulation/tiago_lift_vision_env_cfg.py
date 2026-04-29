"""Vision-only variant of the TIAGo lift task.

Policy input: RGB image from a head camera + proprioception (joint positions/
velocities, last action) — NO privileged object/goal state.
Reward: unchanged from the state-based env (uses simulator ground truth,
which is fine because rewards only run in sim during training).

This follows the standard 'privileged learning' pattern: the reward can
'cheat' because the real robot doesn't need it. Only the policy — which
sees only what a real camera+encoders would see — is deployed.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .tiago_lift_env_cfg import TiagoLiftEnvCfg, TiagoLiftSceneCfg


# Camera resolution. Small is fine — RL works at 64–128 px; bigger just
# costs GPU memory. 84×84 is the DeepMind-style atari default.
CAM_W = 96
CAM_H = 96


@configclass
class VisionObservationsCfg:
    """Policy sees only the RGB image. Single-tensor obs so the CNN extractor
    in skrl's model instantiator can consume it directly."""

    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("head_cam"),
                "data_type": "rgb",
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            # With a single image term, concatenate_terms=True returns the tensor
            # directly (shape N, H, W, C). False returns a dict, which the CNN
            # network instantiator can't handle.
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TiagoLiftVisionEnvCfg(TiagoLiftEnvCfg):
    """Vision-only lift environment. Inherits scene + rewards + actions from the
    state-based env; replaces observations; adds a head camera."""

    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Cameras + 4096 envs is too much GPU memory on most setups. Cap at 1024.
        self.scene.num_envs = min(self.scene.num_envs, 1024)

        # TiledCamera: efficient batched rendering across parallel envs.
        # Attached to head_2_link; offset roughly places it at the xtion camera
        # mount location (on the face of the head). `convention="ros"` means the
        # rot quaternion orients the OPTICAL frame (X right, Y down, Z forward).
        self.scene.head_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_2_link/head_cam",
            update_period=0.0,     # render every env step
            height=CAM_H,
            width=CAM_W,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 5.0),
            ),
            # In head_2_link frame: X forward, Y up, Z lateral (observed earlier).
            # Camera sits at approx xtion mount and looks forward along +X.
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.08, -0.05, 0.0),   # lowered further, roughly at the "chin" of the head
                rot=(0.0, 0.707, 0.0, 0.707),  # optical frame: Z along head_2 +X, Y down
                convention="ros",
            ),
        )


@configclass
class TiagoLiftVisionEnvCfg_PLAY(TiagoLiftVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
