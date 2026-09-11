#!/usr/bin/env python3
"""Baseline vs candidate scatter seam through the real ASE DPA4C EFS path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from ase.calculators.calculator import all_changes
from ase.io import read


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arr_sha(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def eval_calc(calc, atoms) -> dict[str, np.ndarray | float]:
    calc.calculate(atoms=atoms, properties=["energy", "forces", "virial", "stress"], system_changes=all_changes)
    return {
        "energy_eV": float(np.asarray(calc.results["energy"]).reshape(())),
        "forces_eV_per_A": np.asarray(calc.results["forces"], dtype=np.float64).copy(),
        "virial_eV": np.asarray(calc.results["virial"], dtype=np.float64).copy(),
        "stress_eV_per_A3": np.asarray(calc.results["stress"], dtype=np.float64).copy(),
    }


def compare(a, b) -> dict[str, float]:
    return {
        "energy_max_abs": float(abs(a["energy_eV"] - b["energy_eV"])),
        "forces_max_abs": float(np.max(np.abs(a["forces_eV_per_A"] - b["forces_eV_per_A"]))),
        "virial_max_abs": float(np.max(np.abs(a["virial_eV"] - b["virial_eV"]))),
        "stress_max_abs": float(np.max(np.abs(a["stress_eV_per_A3"] - b["stress_eV_per_A3"]))),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--shared-object", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=5)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DP_COMPILE_INFER", "0")
    os.environ.setdefault("DP_TF32_INFER", "0")
    os.environ.setdefault("DP_AMP_INFER", "0")
    result: dict[str, object] = {
        "schema_version": "dpa4c-contest-runtime.efs-demo.v1",
        "status": "FAIL",
        "formal_benchmark": False,
        "timing_scope": "development smoke only; caller-selected warmup/measure; not contest calibrated",
        "stream_contract": "DEFAULT_STREAM_ONLY; producer/consumer stream test not implemented",
        "kernel_trace": {
            "status": "NOT_AVAILABLE",
            "gap": "No profiler/kernel-trace artifact is collected by this prototype.",
        },
        "model_sha256": sha256(a.model),
        "structure_sha256": sha256(a.structure),
        "shared_object_sha256": sha256(a.shared_object),
        "raw_outputs": {},
    }
    try:
        import torch
        from deepmd.calculator import DP
        from runner.runtime_adapter import install

        atoms = read(a.structure, index=0)
        if len(atoms) != 1024 or not bool(np.all(atoms.pbc)):
            raise RuntimeError("demo requires the verified fully-periodic 1024-atom input")
        base_positions = np.asarray(atoms.positions, dtype=np.float64).copy()
        baseline_calc = DP(model=a.model, nlist_backend="auto")
        baseline_cases = []
        candidate_cases = []
        for case, displacement in (("base", 0.0), ("microperturbation", 1.0e-4)):
            atoms.positions = base_positions + displacement * np.sin(
                (np.arange(len(atoms))[:, None] + 1.0)
                * (np.arange(3)[None, :] + 1.0)
                * 0.37
            )
            sync()
            baseline = eval_calc(baseline_calc, atoms)
            baseline_repeat = eval_calc(baseline_calc, atoms)
            baseline_cases.append((case, baseline, baseline_repeat))
        adapter_info = install(a.shared_object)
        candidate_calc = DP(model=a.model, nlist_backend="auto")
        for case, displacement in (("base", 0.0), ("microperturbation", 1.0e-4)):
            atoms.positions = base_positions + displacement * np.sin(
                (np.arange(len(atoms))[:, None] + 1.0)
                * (np.arange(3)[None, :] + 1.0)
                * 0.37
            )
            sync()
            candidate_cases.append((case, eval_calc(candidate_calc, atoms)))
        case_results = {}
        for (name_a, base, baseline_repeat), (name_b, candidate) in zip(baseline_cases, candidate_cases, strict=True):
            assert name_a == name_b
            case_results[name_a] = {
                "diff": compare(base, candidate),
                "baseline_repeat_noise": compare(base, baseline_repeat),
                "baseline": {
                    "energy_eV": base["energy_eV"],
                    "forces_sha256": arr_sha(base["forces_eV_per_A"]),
                    "virial_sha256": arr_sha(base["virial_eV"]),
                    "stress_sha256": arr_sha(base["stress_eV_per_A3"]),
                },
                "candidate": {
                    "energy_eV": candidate["energy_eV"],
                    "forces_sha256": arr_sha(candidate["forces_eV_per_A"]),
                    "virial_sha256": arr_sha(candidate["virial_eV"]),
                    "stress_sha256": arr_sha(candidate["stress_eV_per_A3"]),
                },
            }
            np.savez_compressed(
                a.output.parent / f"{name_a}-raw-efs.npz",
                baseline_energy_eV=np.asarray(base["energy_eV"]),
                baseline_forces_eV_per_A=base["forces_eV_per_A"],
                baseline_virial_eV=base["virial_eV"],
                baseline_stress_eV_per_A3=base["stress_eV_per_A3"],
                baseline_repeat_energy_eV=np.asarray(baseline_repeat["energy_eV"]),
                baseline_repeat_forces_eV_per_A=baseline_repeat["forces_eV_per_A"],
                baseline_repeat_virial_eV=baseline_repeat["virial_eV"],
                baseline_repeat_stress_eV_per_A3=baseline_repeat["stress_eV_per_A3"],
                candidate_energy_eV=np.asarray(candidate["energy_eV"]),
                candidate_forces_eV_per_A=candidate["forces_eV_per_A"],
                candidate_virial_eV=candidate["virial_eV"],
                candidate_stress_eV_per_A3=candidate["stress_eV_per_A3"],
            )
        result["adapter"] = adapter_info
        result["cases"] = case_results

        # Development timing only. The same fixed input is timed after the
        # candidate is installed; no 20+500x3 formal run is started.
        atoms.positions = base_positions
        times = []
        for _ in range(a.warmup):
            sync(); eval_calc(candidate_calc, atoms); sync()
        for _ in range(a.measure):
            sync(); t0 = time.perf_counter(); eval_calc(candidate_calc, atoms); sync(); times.append(time.perf_counter() - t0)
        result["timing"] = {"warmup": a.warmup, "measure": a.measure, "seconds": times, "p50_ms": statistics.median(times) * 1000.0}
        # Temporary development bound only; no contest threshold claim.
        bounds = {"energy_eV": 2.0e-4, "forces_eV_per_A": 5.0e-5, "virial_eV": 2.0e-4, "stress_eV_per_A3": 1.0e-7}
        result["development_tolerances"] = bounds
        diff_bound_keys = {
            "energy_max_abs": bounds["energy_eV"],
            "forces_max_abs": bounds["forces_eV_per_A"],
            "virial_max_abs": bounds["virial_eV"],
            "stress_max_abs": bounds["stress_eV_per_A3"],
        }
        result["status"] = "PASS" if all(
            all(case["diff"][key] <= value for key, value in diff_bound_keys.items())
            for case in case_results.values()
        ) else "FAIL"
        result["note"] = "raw virial is saved and compared directly; stress is not used to reconstruct virial"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
