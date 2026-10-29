#!/usr/bin/env python3
"""Fetch the files needed for demo checks, verifying GitHub blob hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=("agx_arm_sim", "GraspGen"))
    parser.add_argument("--tree", required=True, type=Path)
    args = parser.parse_args()
    tree = json.loads(args.tree.read_text())
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree response is incomplete")
    repository = {"agx_arm_sim": "agilexrobotics/agx_arm_sim", "GraspGen": "vanstrong12138/GraspGen"}[args.project]
    output = ROOT / "third_party" / args.project
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for item in tree["tree"]:
        path = item["path"]
        if item["type"] != "blob":
            continue
        if path in ("README.md", "LICENSE") or (
            args.project == "agx_arm_sim" and path.startswith("mujoco/agilex_arm/agilex_piper/")
        ) or (
            args.project == "GraspGen"
            and path.startswith(("piper_kinematics/", "tensorrtx_yolov8_ros/"))
            and path.endswith((".py", ".cpp", ".hpp", ".h", ".cu", ".launch", ".xml", ".txt", ".md"))
        ):
            files.append(item)

    def fetch(item):
        path, expected = item["path"], item["sha"]
        destination = output / path

        def valid(data):
            return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected

        if destination.is_file() and valid(destination.read_bytes()):
            return path
        url = f"https://raw.githubusercontent.com/{repository}/{tree['sha']}/{quote(path)}"
        for attempt in range(4):
            try:
                with urlopen(url, timeout=40) as response:
                    data = response.read()
                if not valid(data):
                    raise RuntimeError("Blob hash mismatch or incomplete download")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".download")
                temporary.write_bytes(data)
                temporary.replace(destination)
                return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    failures = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        pending = {executor.submit(fetch, item): item for item in files}
        completed = 0
        for future in as_completed(pending):
            completed += 1
            try:
                path = future.result()
                print(f"[{completed}/{len(files)}] verified {path}", flush=True)
            except Exception as exc:
                failures.append({"path": pending[future]["path"], "error": str(exc)})
                print(f"FAILED {pending[future]['path']}: {exc}", flush=True)
    manifest = {"repository": repository, "commit": tree["sha"], "subset": True,
                "files": files, "failures": failures}
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
