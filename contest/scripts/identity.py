#!/usr/bin/env python3
"""Fail-closed Python/package/ELF identity recorder."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-entry", action="store_true")
    parser.add_argument("--expected-artifact", type=Path)
    parser.add_argument("--expected-torch-version")
    parser.add_argument("--expected-torch-root", type=Path)
    args = parser.parse_args()
    result: dict[str, object] = {
        "status": "FAIL",
        "python": str(Path(sys.executable).resolve()),
        "prefix": sys.prefix,
        "cwd": os.getcwd(),
        "sys_path": sys.path,
        "environment": {
            name: os.environ.get(name, "")
            for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH")
        },
    }
    try:
        import deepmd
        import deepmd.lib
        import torch
        from deepmd.calculator import DP  # noqa: F401

        expected = args.expected_prefix.resolve()
        deepmd_path = Path(deepmd.__file__).resolve()
        deepmd_lib_path = Path(deepmd.lib.__file__).resolve()
        if not under(deepmd_path, expected) or not under(deepmd_lib_path, expected):
            raise RuntimeError(
                f"DeepMD import escaped expected prefix {expected}: "
                f"deepmd={deepmd_path}, deepmd.lib={deepmd_lib_path}"
            )
        package_elfs = []
        for path in sorted(deepmd_lib_path.parent.glob("*.so")):
            package_elfs.append(
                {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            )
        if not package_elfs:
            raise RuntimeError("installed deepmd.lib contains no shared objects")
        result.update(
            {
                "deepmd": str(deepmd_path),
                "deepmd_lib": str(deepmd_lib_path),
                "deepmd_elfs": package_elfs,
                "torch": {
                    "version": importlib.metadata.version("torch"),
                    "path": str(Path(torch.__file__).resolve()),
                },
            }
        )
        if (
            args.expected_torch_version
            and result["torch"]["version"] != args.expected_torch_version
        ):
            raise RuntimeError(f"unexpected Torch version: {result['torch']['version']}")
        if args.expected_torch_root and not under(
            Path(torch.__file__), args.expected_torch_root
        ):
            raise RuntimeError(f"Torch escaped frozen root: {torch.__file__}")
        if args.candidate_entry:
            import dpa4c_candidate

            entry = Path(dpa4c_candidate.__file__).resolve()
            if not under(entry, expected):
                raise RuntimeError(f"candidate entry escaped expected prefix: {entry}")
            result["candidate_entry"] = str(entry)
        if args.expected_artifact:
            artifact = args.expected_artifact.resolve()
            if not under(artifact, expected):
                raise RuntimeError(f"candidate artifact escaped expected prefix: {artifact}")
            if not artifact.is_file():
                raise RuntimeError(f"candidate artifact missing: {artifact}")
            result["candidate_artifact"] = {
                "path": str(artifact),
                "sha256": sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        result["status"] = "PASS"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
