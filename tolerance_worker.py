#!/usr/bin/env python3
"""Run one fresh FP32 eager baseline process and save raw E/F/S outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase.calculators.calculator import all_changes
from ase.io import read


def sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def eval_case(calc, atoms) -> dict[str, np.ndarray | float]:
    calc.calculate(
        atoms=atoms,
        properties=["energy", "forces", "virial", "stress"],
        system_changes=all_changes,
    )
    return {
        "energy_eV": np.asarray(calc.results["energy"]).copy(),
        "forces_eV_per_A": np.asarray(calc.results["forces"]).copy(),
        "virial_eV": np.asarray(calc.results["virial"]).copy(),
        "stress_eV_per_A3": np.asarray(calc.results["stress"]).copy(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--amplitude", type=float, required=True)
    p.add_argument("--phase", type=float, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "schema_version": "dpa4c-nano-rc1.tolerance-worker.v1",
        "status": "FAIL",
        "model": str(a.model.resolve()),
        "structure": str(a.structure.resolve()),
        "amplitude": a.amplitude,
        "phase": a.phase,
        "execution": "single fresh FP32 eager PPU baseline process",
    }
    try:
        os.environ["DP_COMPILE_INFER"] = "0"
        os.environ["DP_TF32_INFER"] = "0"
        os.environ["DP_AMP_INFER"] = "0"
        import torch
        from deepmd.calculator import DP

        if not torch.cuda.is_available():
            raise RuntimeError("PPU/CUDA device unavailable")
        torch.set_default_dtype(torch.float32)
        atoms = read(a.structure, index=0)
        if len(atoms) != 1024 or not bool(np.all(atoms.pbc)):
            raise RuntimeError("expected fully periodic 1024-atom structure")
        positions = np.asarray(atoms.positions, dtype=np.float64).copy()
        phase = float(a.phase)
        displacement = a.amplitude * np.sin(
            (np.arange(len(atoms))[:, None] + 1.0)
            * (np.arange(3)[None, :] + 1.0)
            * (0.37 + phase)
        )
        calc = DP(model=a.model, nlist_backend="auto")
        model = calc.dp.deep_eval._dpmodel
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
        if parameter_dtypes != ["torch.float32"]:
            raise RuntimeError(f"expected FP32 model parameters, got {parameter_dtypes}")
        atoms.positions = positions
        sync()
        anchor = eval_case(calc, atoms)
        atoms.positions = positions + displacement
        unique = eval_case(calc, atoms)
        sync()
        anchor_arrays = {key: np.asarray(value) for key, value in anchor.items()}
        unique_arrays = {key: np.asarray(value) for key, value in unique.items()}
        np.savez_compressed(a.output.with_name("anchor.npz"), **anchor_arrays)
        np.savez_compressed(a.output.with_name("unique.npz"), **unique_arrays)
        status["device"] = {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "parameter_dtypes": parameter_dtypes,
        }
        status["arrays"] = {
            case: {
                key: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "finite": bool(np.isfinite(value).all()),
                }
                for key, value in arrays.items()
            }
            for case, arrays in (("anchor", anchor_arrays), ("unique", unique_arrays))
        }
        status["status"] = "PASS"
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
    a.output.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
