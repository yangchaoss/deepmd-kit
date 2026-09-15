#!/usr/bin/env python3
"""One fresh process for one route of the public paired benchmark."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys
import time
from pathlib import Path

import numpy as np


PROTOCOL = "dpa4c-ppu-contest.public-benchmark.v2"
STREAM = None


def send(value: object) -> None:
    payload = pickle.dumps(value, protocol=5)
    STREAM.write(struct.pack(">Q", len(payload)))
    STREAM.write(payload)
    STREAM.flush()


def receive() -> object | None:
    header = STREAM.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise RuntimeError("truncated protocol header")
    size = struct.unpack(">Q", header)[0]
    payload = STREAM.read(size)
    if len(payload) != size:
        raise RuntimeError("truncated protocol payload")
    return pickle.loads(payload)


def sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory(torch) -> dict[str, int | None]:
    result = {"allocated_bytes": None, "reserved_bytes": None,
              "vram_free_bytes": None, "vram_total_bytes": None}
    try:
        result["allocated_bytes"] = int(torch.cuda.memory_allocated())
        result["reserved_bytes"] = int(torch.cuda.memory_reserved())
        free, total = torch.cuda.mem_get_info()
        result["vram_free_bytes"] = int(free)
        result["vram_total_bytes"] = int(total)
    except Exception:
        pass
    return result


def normalize(values: dict[str, object]) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(values["energy"]),
        np.asarray(values["forces"]).copy(),
        np.asarray(values["virial"]).copy(),
        np.asarray(values["stress"]).copy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("reference", "baseline", "candidate"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measure", type=int, required=True)
    parser.add_argument("--protocol-fd", type=int, required=True)
    args = parser.parse_args()
    global STREAM
    STREAM = os.fdopen(args.protocol_fd, "r+b", buffering=0)
    for name, value in (("DP_COMPILE_INFER", "0"), ("DP_TF32_INFER", "0"),
                        ("DP_AMP_INFER", "0")):
        os.environ.setdefault(name, value)
    summary: dict[str, object] = {
        "schema_version": PROTOCOL, "route": args.route, "pid": os.getpid(),
        "status": "FAIL", "warmup_count": args.warmup,
        "measure_count": args.measure,
        "timing_boundary": (
            "host positions/cell materialized and assigned before timer; "
            "device sync + evaluate + device sync + host E/F/virial/stress readiness inside timer"
        ),
    }
    try:
        import torch
        from ase.io import read
        from ase.calculators.calculator import all_changes

        if args.route == "candidate":
            from dpa4c_candidate import create_session
            evaluate = create_session(model=str(args.model)).evaluate
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
            raise RuntimeError("public benchmark requires the fixed periodic 1024-atom structure")
        summary.update({
            "python": str(Path(sys.executable).resolve()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        })
        send({"ready": True, "pid": os.getpid(), "route": args.route})
        warmup_times: list[float] = []
        measured_times: list[float] = []
        energies: list[float] = []
        forces: list[np.ndarray] = []
        virials: list[np.ndarray] = []
        stresses: list[np.ndarray] = []
        samples: list[dict[str, object]] = []
        dtypes = None
        while True:
            message = receive()
            if not isinstance(message, dict):
                raise RuntimeError("invalid or truncated parent message")
            if message.get("kind") == "finish":
                break
            if message.get("kind") != "frame":
                raise RuntimeError("unexpected protocol message")
            index = int(message["index"])
            is_warmup = bool(message["warmup"])
            positions = np.asarray(message["positions"], dtype=np.float64)
            cell = np.asarray(message["cell"], dtype=np.float64)
            if positions.shape != (1024, 3) or cell.shape != (3, 3):
                raise RuntimeError(f"invalid frame shape at {index}")
            atoms.positions = positions
            atoms.set_cell(cell, scale_atoms=False)
            sync(torch)
            start = time.perf_counter()
            energy, force, virial, stress = normalize(evaluate(atoms))
            sync(torch)
            # numpy conversion/copy above makes host outputs ready inside the timer.
            elapsed = time.perf_counter() - start
            if dtypes is None:
                dtypes = {"energy": str(energy.dtype), "forces": str(force.dtype),
                          "virial": str(virial.dtype), "stress": str(stress.dtype)}
            if is_warmup:
                warmup_times.append(elapsed)
            else:
                measured_times.append(elapsed)
                energies.append(float(energy.reshape(())))
                forces.append(force)
                virials.append(virial)
                stresses.append(stress)
            sample: dict[str, object] = memory(torch)
            sample.update({"index": index, "warmup": is_warmup})
            samples.append(sample)
            send({"ok": True, "index": index})
        if len(warmup_times) != args.warmup or len(measured_times) != args.measure:
            raise RuntimeError("frame count mismatch")
        np.savez_compressed(
            args.output,
            measured_energy_eV=np.asarray(energies, dtype=np.float64),
            measured_forces=np.stack(forces), measured_virial=np.stack(virials),
            measured_stress=np.stack(stresses),
            warmup_latencies_s=np.asarray(warmup_times, dtype=np.float64),
            measured_latencies_s=np.asarray(measured_times, dtype=np.float64),
        )
        measured = np.asarray(measured_times, dtype=np.float64)
        summary.update({
            "status": "PASS", "result_dtypes": dtypes, "memory_samples": samples,
            "latency": {
                "mean_s": float(np.mean(measured)),
                "p50_s": float(np.quantile(measured, 0.50)),
                "p90_s": float(np.quantile(measured, 0.90)),
                "p99_s": float(np.quantile(measured, 0.99)),
                "cv": float(np.std(measured, ddof=1) / np.mean(measured)),
                "throughput_evals_per_s": float(1.0 / np.mean(measured)),
                "throughput_atoms_per_s": float(1024.0 / np.mean(measured)),
            },
        })
        Path(str(args.output) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        send({"finished": True, "status": "PASS", "summary": summary})
        return 0
    except Exception as exc:
        summary.update({"error_type": type(exc).__name__, "error": str(exc)})
        Path(str(args.output) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        try:
            send({"finished": True, "status": "FAIL", "error": str(exc)})
        except Exception:
            pass
        print(json.dumps(summary), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
