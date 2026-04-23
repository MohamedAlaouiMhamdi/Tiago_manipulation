"""One-shot: bake the TIAGo HSR color scheme into the USD.

Strategy: the per-link Xform has a `rel material:binding` authored as
strongerThanDescendants pointing at e.g. /tiago/Looks/material_DarkGrey,
which overrides anything we'd author on the mesh or its DefaultMaterial.
So we (a) create three new HSR materials under /tiago/Looks/ and (b)
rewrite the Xform-level binding on each link to point at the HSR material
that matches the link's role (body -> white, arm/gripper -> black,
wheels -> very dark).

Usage:
    python scripts/recolor_tiago_usd.py \
        source/Tiago_manipulation/Tiago_manipulation/assets/usd_omni_colored/configuration/tiago_base.usd
"""

import argparse
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdShade


WHITE = (0.95, 0.95, 0.95)
BLACK = (0.0, 0.0, 0.0)

# Materials that count as "dark" in the original USD — anything bound to one of
# these gets repainted black; everything else becomes white.
DARK_ORIGINALS = {
    "material_DarkGrey",
    "material_FlatBlack",
    "material_Black",
    "material_404040",
}


def ensure_hsr_materials(stage: Usd.Stage, looks_path: str) -> dict[str, str]:
    """Create (or replace) the HSR OmniPBR materials. Returns dict role -> path."""
    defs = {
        "white": ("HSR_White", WHITE, 0.6),
        "black": ("HSR_Black", BLACK, 0.95),  # very rough so it doesn't highlight
    }
    out: dict[str, str] = {}
    for role, (name, rgb, roughness) in defs.items():
        mat_path = f"{looks_path}/{name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader_path = f"{mat_path}/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        # explicitly disable emission (OmniPBR defaults emissive_color=(1,1,1) intensity=10000)
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(False)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(0.0)
        # kill specular bounce for dark surfaces
        shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0 if role == "black" else 0.5)
        shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader_out)
        mat.CreateDisplacementOutput("mdl").ConnectToSource(shader_out)
        mat.CreateVolumeOutput("mdl").ConnectToSource(shader_out)
        out[role] = mat_path
    return out


def recolor(usd_path: Path) -> None:
    if not usd_path.exists():
        sys.exit(f"USD not found: {usd_path}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        sys.exit(f"Failed to open stage: {usd_path}")

    # Find /tiago/Looks or similar location for the named materials
    looks_candidates = [p for p in stage.Traverse()
                        if p.GetTypeName() == "Scope" and p.GetName() == "Looks"
                        and "/tiago/" in str(p.GetPath()).rsplit("/", 1)[0] + "/"]
    if not looks_candidates:
        # fallback: find by path
        fallback = stage.GetPrimAtPath("/tiago/Looks")
        if fallback.IsValid():
            looks_candidates = [fallback]
    if not looks_candidates:
        sys.exit("Could not find /tiago/Looks — is this the right USD layer?")
    looks_path = str(looks_candidates[0].GetPath())

    hsr = ensure_hsr_materials(stage, looks_path)
    print(f"HSR materials under {looks_path}: {hsr}")

    rebound = {"white": 0, "black": 0, "kept": 0}
    for prim in stage.Traverse():
        rel = prim.GetRelationship("material:binding")
        if not rel or not rel.IsValid():
            continue
        if not rel.HasAuthoredTargets():
            continue
        path_lower = str(prim.GetPath()).lower()
        # Preserve the original PAL-authored material on wheels/casters so they
        # keep their factory dark-grey look.
        if "wheel" in path_lower or "caster" in path_lower or "tire" in path_lower:
            rebound["kept"] += 1
            continue
        targets = rel.GetTargets()
        orig_mat_name = targets[0].name if targets else ""
        role = "black" if orig_mat_name in DARK_ORIGINALS else "white"
        rel.SetTargets([Sdf.Path(hsr[role])])
        UsdShade.MaterialBindingAPI(prim).SetMaterialBindingStrength(
            rel, UsdShade.Tokens.strongerThanDescendants
        )
        rebound[role] += 1

    stage.Save()
    print(
        f"Recolored {usd_path.name}: rebound {rebound['white'] + rebound['black']} bindings "
        f"(white={rebound['white']} black={rebound['black']}, kept-original={rebound['kept']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake HSR colors into a TIAGo USD in place.")
    parser.add_argument("usd_path", type=Path, help="Path to the USD to recolor in-place.")
    args = parser.parse_args()
    recolor(args.usd_path)
