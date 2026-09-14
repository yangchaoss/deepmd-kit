#!/usr/bin/env python3
"""Validate and bind a freshly built custom extension without source shadowing."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = json.loads(args.build_json.read_text(encoding="utf-8"))
    objects = build.get("shared_objects", [])
    if build.get("status") != "PASS" or len(objects) != 1:
        raise SystemExit("build record must contain one successful shared object")
    shared_object = Path(objects[0]["path"]).resolve()
    if not shared_object.is_file() or sha256(shared_object) != objects[0]["sha256"]:
        raise SystemExit("shared object does not match the build record")
    import deepmd
    import torch

    torch.ops.load_library(str(shared_object))
    if not hasattr(torch.ops.dpa4c_contest, "edge_force_virial"):
        raise SystemExit("candidate operator was not registered")
    package_root = Path(deepmd.__file__).resolve().parent
    deepmd_elfs = [
        {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(package_root.rglob("*.so"))
    ]
    repository_root = Path(__file__).resolve().parents[4]
    result = {
        "schema_version": "dpa4c-contest-runtime.install-record.v1",
        "status": "PASS",
        "python": {"executable": sys.executable, "prefix": sys.prefix},
        "deepmd": {"module": deepmd.__file__, "version": getattr(deepmd, "__version__", None), "elfs": deepmd_elfs},
        "shared_object": {"path": str(shared_object), "sha256": sha256(shared_object), "size": shared_object.stat().st_size},
        "operator": "dpa4c_contest::edge_force_virial",
        "source_checkout_on_sys_path": any(
            entry and Path(entry).resolve() == repository_root for entry in sys.path
        ),
    }
    if result["source_checkout_on_sys_path"]:
        raise SystemExit("source checkout shadows the installed runtime")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
