#!/usr/bin/env python3
"""Read the composed upstream USD without starting Isaac Sim."""
import json
from pathlib import Path

from pxr import Usd

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "third_party/IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf/piper/usd/piper.usd"
OUTPUT = ROOT / "records/isaac_stack/piper_usd_audit.json"


def main():
    stage = Usd.Stage.Open(str(MODEL))
    if stage is None:
        raise RuntimeError(f"Could not open {MODEL}")
    result = {"model": str(MODEL), "layers": [str(x.identifier) for x in stage.GetUsedLayers()],
              "authored_contact_friction": [], "gripper_joint_properties": [], "physics_material_bindings": []}
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))
    result["prim_count"] = len(prims)
    for prim in prims:
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if not attribute.HasAuthoredValueOpinion():
                continue
            contact_friction = "friction" in name.lower() and "jointfriction" not in name.lower()
            gripper_property = "gripper_joint" in str(prim.GetPath()) and (
                name.startswith("drive:") or name in ("physics:axis", "physics:lowerLimit", "physics:upperLimit"))
            if not contact_friction and not gripper_property:
                continue
            value = attribute.Get()
            record = {"prim": str(prim.GetPath()), "attribute": name,
                      "value": value if isinstance(value, (bool, int, float, str)) else str(value)}
            if contact_friction:
                result["authored_contact_friction"].append(record)
            if gripper_property:
                result["gripper_joint_properties"].append(record)
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith("material:binding:physics"):
                result["physics_material_bindings"].append({
                    "prim": str(prim.GetPath()), "targets": [str(x) for x in relationship.GetTargets()]})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
