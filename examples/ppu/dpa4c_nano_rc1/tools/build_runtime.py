#!/usr/bin/env python3
"""Official fixed build wrapper; candidate setup.py/build.sh is never executed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "candidate" / "dpa4c_contest_ops.cu",
    )
    parser.add_argument("--build", type=Path, default=root / "build")
    parser.add_argument("--result", type=Path, default=root / "evidence" / "build.json")
    args = parser.parse_args()
    allowed = (root / "candidate" / "dpa4c_contest_ops.cu").resolve()
    source = args.source.resolve()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": "dpa4c-contest-runtime.build.v1",
        "status": "FAIL",
        "source": str(source),
        "source_sha256": sha256(source) if source.exists() else None,
        "allowed_source": str(allowed),
        "candidate_setup_executed": False,
        "build_dir": str(args.build.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "ppu_sdk": os.environ.get("PPU_SDK"),
    }
    try:
        if source != allowed:
            raise RuntimeError(f"source is outside the official whitelist: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        import torch
        from torch.utils.cpp_extension import load

        result["torch"] = {
            "version": torch.__version__,
            "file": torch.__file__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compiled_with_cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        }
        args.build.mkdir(parents=True, exist_ok=True)
        load(
            name="dpa4c_contest_ops",
            sources=[str(source)],
            build_directory=str(args.build.resolve()),
            extra_cflags=["-O0", "-g0"],
            extra_cuda_cflags=["-O0", "-g0"],
            is_python_module=False,
            with_cuda=True,
            verbose=True,
        )
        so_files = sorted(args.build.rglob("*.so"))
        if not so_files:
            raise RuntimeError("fixed build completed without a shared object")
        result["shared_objects"] = [
            {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}
            for path in so_files
        ]
        result["status"] = "PASS"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
