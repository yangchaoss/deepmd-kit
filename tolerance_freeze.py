#!/usr/bin/env python3
"""Freeze baseline repeatability into an auditable development tolerance report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


FIELDS = ("energy_eV", "forces_eV_per_A", "virial_eV", "stress_eV_per_A3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def pairwise_max(values: list[np.ndarray]) -> float:
    return max(max_abs(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--source-tree", required=True)
    p.add_argument("--worker-count", type=int, default=3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    worker_root = a.output / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": "dpa4c-nano-rc1.tolerance-freeze.v1",
        "status": "BLOCKED",
        "gate": "TOLERANCE_FREEZE",
        "execution_scope": "DEVELOPMENT_SANDBOX",
        "baseline": {
            "route": "deepmd.calculator.DP(.pt, nlist_backend=auto), eager, FP32, single PPU",
            "model": str(a.model.resolve()),
            "model_sha256": sha256(a.model),
            "structure": str(a.structure.resolve()),
            "structure_sha256": sha256(a.structure),
            "source_commit": a.source_commit,
            "source_tree": a.source_tree,
        },
        "workers": [],
        "tolerance_formula": {
            "atol": "max(4 * max_pairwise_anchor_noise, 64 * dtype_eps * max(1, max_abs_anchor))",
            "rtol": "64 * dtype_eps",
            "floating_slack": "64 * dtype_eps * max(1, max_abs_anchor)",
            "note": "Derived from three fresh baseline processes; not copied from development probe thresholds.",
        },
    }
    try:
        if a.worker_count < 3:
            raise ValueError("worker-count must be at least 3")
        worker = Path(__file__).resolve().parent / "tolerance_worker.py"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "DP_COMPILE_INFER": "0",
                "DP_TF32_INFER": "0",
                "DP_AMP_INFER": "0",
            }
        )
        amplitudes = (0.0, 1.0e-4, 2.0e-4)
        phases = (0.0, 0.11, 0.23)
        for index in range(a.worker_count):
            worker_dir = worker_root / f"worker-{index:02d}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            result_path = worker_dir / "worker.json"
            command = [
                sys.executable,
                str(worker),
                "--model",
                str(a.model),
                "--structure",
                str(a.structure),
                "--output",
                str(result_path),
                "--amplitude",
                str(amplitudes[index % len(amplitudes)]),
                "--phase",
                str(phases[index % len(phases)]),
            ]
            completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
            (worker_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (worker_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            worker_info: dict[str, object] = {
                "index": index,
                "returncode": completed.returncode,
                "result": str(result_path),
                "stdout": str(worker_dir / "stdout.log"),
                "stderr": str(worker_dir / "stderr.log"),
            }
            if result_path.exists():
                worker_info["status"] = json.loads(result_path.read_text(encoding="utf-8")).get("status")
            report["workers"].append(worker_info)
            if completed.returncode != 0 or worker_info.get("status") != "PASS":
                raise RuntimeError(f"fresh baseline worker {index} failed")

        anchor_data = []
        unique_data = []
        for index in range(a.worker_count):
            worker_dir = worker_root / f"worker-{index:02d}"
            anchor_data.append(np.load(worker_dir / "anchor.npz"))
            unique_data.append(np.load(worker_dir / "unique.npz"))
        all_finite = True
        shapes = {}
        for name, datasets in (("anchor", anchor_data), ("unique", unique_data)):
            shapes[name] = {}
            for field in FIELDS:
                arrays = [np.asarray(dataset[field]) for dataset in datasets]
                shapes[name][field] = list(arrays[0].shape)
                all_finite = all_finite and all(bool(np.isfinite(array).all()) for array in arrays)
                if any(array.shape != arrays[0].shape for array in arrays):
                    raise RuntimeError(f"shape mismatch for {name}/{field}")
        if not all_finite:
            raise RuntimeError("non-finite baseline E/F/S output")

        tolerance = {}
        noise = {}
        for field in FIELDS:
            arrays = [np.asarray(dataset[field]) for dataset in anchor_data]
            dtype = np.result_type(*[array.dtype for array in arrays])
            if not np.issubdtype(dtype, np.floating):
                raise RuntimeError(f"non-floating dtype for {field}: {dtype}")
            eps = float(np.finfo(dtype).eps)
            scale = float(max(1.0, max(np.max(np.abs(array)) for array in arrays)))
            observed = pairwise_max(arrays)
            slack = 64.0 * eps * scale
            noise[field] = {
                "dtype": str(dtype),
                "max_pairwise_anchor_abs": observed,
                "max_abs_anchor": scale,
                "dtype_eps": eps,
            }
            tolerance[field] = {
                "atol": max(4.0 * observed, slack),
                "rtol": 64.0 * eps,
                "floating_slack": slack,
            }
        report["all_raw_finite"] = all_finite
        report["shapes"] = shapes
        report["anchor_repeatability"] = noise
        report["frozen_tolerance"] = tolerance
        report["status"] = "PASS"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)

    report_path = a.output / "tolerance-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = [
        "# DPA4C Nano rc1 baseline tolerance freeze",
        "",
        f"- Status: **{report['status']}**",
        "- Scope: `DEVELOPMENT_SANDBOX` (PPU; not Frozen Runtime Image)",
        f"- Source commit: `{a.source_commit}`",
        f"- Source tree: `{a.source_tree}`",
        f"- Model SHA256: `{report['baseline']['model_sha256']}`",
        f"- Structure SHA256: `{report['baseline']['structure_sha256']}`",
        "",
        "Tolerances are derived from three fresh-process baseline anchor runs using the formula in the JSON report. They are not copied from the prior development probe.",
    ]
    (a.output / "tolerance-report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    checksum_paths = sorted(path for path in a.output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (a.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(a.output)}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
