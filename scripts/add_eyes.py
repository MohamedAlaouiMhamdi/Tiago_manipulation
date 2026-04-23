"""Add two black sphere 'eyes' to the TIAGo head in the USD.

The xtion camera mesh doesn't import cleanly via the URDF converter, so this
attaches two small decorative spheres under head_2_link (positioned where the
camera lenses sit on the real HSR).

Usage:
    python scripts/add_eyes.py \
        source/Tiago_manipulation/Tiago_manipulation/assets/usd_cam_colored/configuration/tiago_base.usd
"""

import argparse
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


# Positions in head_2_link frame (meters).
# Observed from previous attempts: Y=+0.125 is the TOP of the head, not the front.
# So X = forward, Y = up/down, Z = left/right. Front face is at X=+0.108.
# Put eyes just past the front face at upper-mid height, ~5cm apart.
EYE_RADIUS = 0.014
EYE_OFFSETS = {
    "eye_left":  (0.11, 0.04, +0.025),
    "eye_right": (0.11, 0.04, -0.025),
}


def ensure_black_material(stage: Usd.Stage, looks_path: str) -> str:
    mat_path = f"{looks_path}/HSR_EyeBlack"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = f"{mat_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.2)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.5)  # a little shine
    shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader_out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader_out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader_out)
    return mat_path


def add_eyes(usd_path: Path) -> None:
    if not usd_path.exists():
        sys.exit(f"USD not found: {usd_path}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        sys.exit(f"Failed to open stage: {usd_path}")

    # Locate head_2_link under the visual layer (/visuals/head_2_link)
    head = stage.GetPrimAtPath("/visuals/head_2_link")
    if not head.IsValid():
        sys.exit("Could not find /visuals/head_2_link")

    # Create black material under /tiago/Looks
    looks = stage.GetPrimAtPath("/tiago/Looks")
    if not looks.IsValid():
        sys.exit("Could not find /tiago/Looks")
    mat_path = ensure_black_material(stage, str(looks.GetPath()))

    # Add the eyes
    for name, (x, y, z) in EYE_OFFSETS.items():
        path = f"{head.GetPath()}/{name}"
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.GetRadiusAttr().Set(EYE_RADIUS)
        # Pure black needs to be visible -> make it a proper xformable prim
        xf = UsdGeom.Xformable(sphere)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        print(f"Added eye: {path}  at ({x}, {y}, {z})  r={EYE_RADIUS}")

    stage.Save()
    print(f"\nSaved {usd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add eye spheres to head_2_link in a TIAGo USD.")
    parser.add_argument("usd_path", type=Path)
    args = parser.parse_args()
    add_eyes(args.usd_path)
