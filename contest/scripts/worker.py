#!/usr/bin/env python3
"""Organization-owned one-process E/F/virial/stress smoke worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np
from ase.calculators.calculator import all_changes
from ase.io import read


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def normalize(results: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "energy_eV": np.asarray(results["energy"], dtype=np.float64).reshape(()),
        "forces_eV_per_A": np.asarray(results["forces"], dtype=np.float64),
        "virial_eV": np.asarray(results["virial"], dtype=np.float64),
        "stress_eV_per_A3": np.asarray(results["stress"], dtype=np.float64),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result: dict[str, object] = {"status": "FAIL", "mode": args.mode}
    try:
        os.environ.setdefault("DP_COMPILE_INFER", "0")
        os.environ.setdefault("DP_TF32_INFER", "0")
        os.environ.setdefault("DP_AMP_INFER", "0")
        import deepmd
        import deepmd.lib
        import torch

        if args.mode == "candidate":
            from dpa4c_candidate import create_session

            session = create_session(model=str(args.model))
            evaluate = session.evaluate
        else:
            from deepmd.calculator import DP

            calculator = DP(model=args.model, nlist_backend="auto")

            def evaluate(atoms):
                calculator.calculate(
                    atoms=atoms,
                    properties=["energy", "forces", "virial", "stress"],
                    system_changes=all_changes,
                )
                return dict(calculator.results)

        atoms = read(args.structure, index=0)
        if len(atoms) != 1024 or not bool(np.all(atoms.pbc)):
            raise RuntimeError("public smoke requires the fixed periodic 1024-atom structure")
        base = np.asarray(atoms.positions, dtype=np.float64).copy()
        frames: dict[str, dict[str, np.ndarray]] = {}
        for name, displacement in (("base", 0.0), ("microperturbation", 1.0e-4)):
            atoms.positions = base + displacement * np.sin(
                (np.arange(len(atoms))[:, None] + 1.0)
                * (np.arange(3)[None, :] + 1.0)
                * 0.37
            )
            sync()
            frames[name] = normalize(evaluate(atoms))
            sync()
        arrays = {
            f"{frame}__{field}": value
            for frame, values in frames.items()
            for field, value in values.items()
        }
        np.savez_compressed(args.output / "efs.npz", **arrays)
        checks = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "isfinite": bool(np.isfinite(value).all()),
            }
            for key, value in arrays.items()
        }
        if not all(item["isfinite"] for item in checks.values()):
            raise RuntimeError("non-finite E/F/virial/stress output")
        maps = []
        maps_path = Path("/proc/self/maps")
        if maps_path.is_file():
            for line in maps_path.read_text(errors="replace").splitlines():
                path = line.split()[-1] if "/" in line else ""
                if path and ("deepmd" in path or "dpa4c" in path):
                    maps.append(path)
        result.update(
            {
                "status": "PASS",
                "model_sha256": sha256(args.model),
                "structure_sha256": sha256(args.structure),
                "deepmd": str(Path(deepmd.__file__).resolve()),
                "deepmd_lib": str(Path(deepmd.lib.__file__).resolve()),
                "torch": {"version": torch.__version__, "device": torch.cuda.get_device_name(0)},
                "outputs": checks,
                "loaded_maps": sorted(set(maps)),
            }
        )
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    (args.output / "worker.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
