"""Spawn the TIAGo USD and sweep each arm + gripper joint through its full range.

Cycles through arm_1..arm_7, then the gripper (both fingers together).
Each joint gets SECONDS_PER_JOINT seconds of a single-period sine wave
that reaches both the lower and upper USD limits.

Run:
    ./IsaacLab/isaaclab.sh -p scripts/sanity_check_tiago.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TIAGo per-joint full-range sanity check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Gf, Sdf, Usd, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

TIAGO_USD = (
    "/home/mohamed/Tiago_manipulation/Tiago_manipulation/"
    "source/Tiago_manipulation/Tiago_manipulation/assets/usd_final/tiago.usd"
)

SECONDS_PER_JOINT = 5.0

# Joints to sweep, in order. "gripper" is a pseudo-entry that drives both fingers together.
SWEEP_ORDER = [
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint",
    "gripper",
]


def paint_hsr(stage: Usd.Stage, root: str = "/World/Tiago") -> None:
    """Apply the TIAGo HSR color scheme by populating each mesh's sibling
    DefaultMaterial with a UsdPreviewSurface shader."""

    def color_for(prim_path: str) -> tuple[float, float, float]:
        p = prim_path.lower()
        if "wheel" in p or "caster" in p or "tire" in p:
            return (0.05, 0.05, 0.05)
        if "arm_" in p or "gripper" in p or "wrist" in p:
            return (0.10, 0.10, 0.10)
        return (0.92, 0.92, 0.92)

    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        print(f"paint_hsr: root prim {root} not found!")
        return

    # De-instance so we can author overrides on the mesh subgraph.
    deinstanced = 0
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            deinstanced += 1
    if deinstanced:
        print(f"paint_hsr: de-instanced {deinstanced} prims under {root}")

    painted = {"white": 0, "arm": 0, "wheel": 0}
    missing_default_mat = 0
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "Mesh":
            continue
        parent = prim.GetParent()
        default_mat_path = f"{parent.GetPath()}/Looks/DefaultMaterial"
        default_mat_prim = stage.GetPrimAtPath(default_mat_path)
        if not default_mat_prim.IsValid():
            missing_default_mat += 1
            continue

        rgb = color_for(str(prim.GetPath()))
        mat = UsdShade.Material(default_mat_prim)
        shader = UsdShade.Shader.Define(stage, f"{default_mat_path}/HSRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        if rgb == (0.05, 0.05, 0.05):
            painted["wheel"] += 1
        elif rgb == (0.10, 0.10, 0.10):
            painted["arm"] += 1
        else:
            painted["white"] += 1
    print(f"paint_hsr: painted {sum(painted.values())} meshes via DefaultMaterial "
          f"(white={painted['white']} arm={painted['arm']} wheel={painted['wheel']}, "
          f"missing_default_mat={missing_default_mat})")


def design_scene() -> Articulation:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    cfg = ArticulationCfg(
        prim_path="/World/Tiago",
        spawn=sim_utils.UsdFileCfg(usd_path=TIAGO_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=None, damping=None
            ),
        },
    )
    return Articulation(cfg=cfg)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 120))
    sim.set_camera_view(eye=(2.5, 2.5, 1.8), target=(0.0, 0.0, 0.8))

    robot = design_scene()
    sim.reset()

    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
    limits = robot.data.soft_joint_pos_limits[0]  # (num_joints, 2) -> (lo, hi)

    def lims_for(name: str) -> tuple[float, float]:
        i = name_to_idx[name]
        lo = limits[i, 0].item()
        hi = limits[i, 1].item()
        return lo, hi

    print("=" * 70)
    print("Full-range sweep plan:")
    for n in SWEEP_ORDER:
        if n == "gripper":
            lo, hi = lims_for("gripper_left_finger_joint")
            print(f"  gripper (both fingers)      range=[{lo:+.3f}, {hi:+.3f}] m")
        else:
            lo, hi = lims_for(n)
            print(f"  {n:24s}  range=[{lo:+.3f}, {hi:+.3f}] rad")
    print("=" * 70)

    dt = sim.get_physics_dt()
    default_q = robot.data.default_joint_pos.clone()
    phase_steps = int(SECONDS_PER_JOINT / dt)

    step = 0
    current = None
    while simulation_app.is_running():
        which = (step // phase_steps) % len(SWEEP_ORDER)
        t_in_phase = (step % phase_steps) * dt
        active = SWEEP_ORDER[which]

        if active != current:
            print(f"\n>>> moving {active}")
            current = active

        target = default_q.clone()
        s = math.sin(2 * math.pi * t_in_phase / SECONDS_PER_JOINT)  # -1..1

        if active == "gripper":
            lo, hi = lims_for("gripper_left_finger_joint")
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            val = center + half * s
            for name in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
                target[:, name_to_idx[name]] = val
            probe_idx = name_to_idx["gripper_left_finger_joint"]
            probe_label = "gripper_L"
        else:
            lo, hi = lims_for(active)
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            target[:, name_to_idx[active]] = center + half * s
            probe_idx = name_to_idx[active]
            probe_label = active

        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if step % int(0.5 / dt) == 0:
            q = robot.data.joint_pos[0, probe_idx].item()
            tgt = target[0, probe_idx].item()
            print(f"  t={t_in_phase:4.1f}s  {probe_label:26s} target={tgt:+.3f}  actual={q:+.3f}")
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
