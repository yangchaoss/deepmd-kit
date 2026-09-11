#!/usr/bin/env python3
"""Bounded environment and input gate for the private prototype."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": "dpa4c-contest-runtime.preflight.v1",
        "status": "FAIL",
        "root": str(root),
        "model": str(a.model.resolve()),
        "structure": str(a.structure.resolve()),
        "source": str(
            (root / "candidate" / "dpa4c_contest_ops.cu").resolve()
        ),
        "git_head": git_value(root, "rev-parse", "HEAD"),
        "git_source_snapshot": git_value(root, "rev-parse", "HEAD^{tree}"),
        "checks": {},
    }
    try:
        source = root / "candidate" / "dpa4c_contest_ops.cu"
        checks = result["checks"]
        assert isinstance(checks, dict)
        checks["source_whitelist"] = (
            source.is_file()
            and source.resolve().parent == (root / "candidate").resolve()
        )
        checks["no_model_in_distribution"] = not any(root.rglob("*.pt")) and not any(root.rglob("*.pt2"))
        checks["no_secret_like_files"] = not any(
            part.name.lower() in {".env", "credentials", "token", "ackey", "api_key"}
            for part in root.rglob("*")
            if part.is_file()
        )
        checks["model_exists"] = a.model.is_file()
        checks["structure_exists"] = a.structure.is_file()
        checks["required_env"] = bool(os.environ.get("PPU_SDK")) and bool(os.environ.get("CUDA_HOME"))
        checks["fixed_mode"] = os.environ.get("DPA4C_CONTEST_FORMAL", "0") != "1"
        if not all(bool(value) for value in checks.values()):
            raise RuntimeError("one or more bounded preflight checks failed")
        result["source_sha256"] = sha256(source)
        result["model_sha256"] = sha256(a.model)
        result["structure_sha256"] = sha256(a.structure)
        result["execution_scope"] = "PPU small probe + 1024-atom ASE development smoke only"
        result["status"] = "PASS"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
