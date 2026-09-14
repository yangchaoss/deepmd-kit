#!/usr/bin/env python3
"""Public Nano/1024 FP32 paired self-test benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import socket
import statistics
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


PROTOCOL = "dpa4c-ppu-contest.public-benchmark.v1"
PUBLIC_SEEDS = (510421, 620531, 730637)
PAIR_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"),
               ("baseline", "candidate"))


class RouteFailure(RuntimeError):
    def __init__(self, route: str, message: str):
        super().__init__(message)
        self.status = "BENCHMARK_INVALID" if route in {"reference", "baseline"} else "CANDIDATE_INVALID"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(positions: np.ndarray, cell: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(positions, dtype=np.float64).tobytes(order="C"))
    digest.update(np.asarray(cell, dtype=np.float64).tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def send(stream, value: object) -> None:
    payload = pickle.dumps(value, protocol=5)
    stream.write(struct.pack(">Q", len(payload)))
    stream.write(payload)
    stream.flush()


def receive(stream) -> object:
    header = stream.read(8)
    if len(header) != 8:
        raise RuntimeError("truncated worker response")
    size = struct.unpack(">Q", header)[0]
    payload = stream.read(size)
    if len(payload) != size:
        raise RuntimeError("truncated worker payload")
    return pickle.loads(payload)


def load_atoms(structure: Path):
    from ase.io import read

    return read(structure, index=0)


def generate_sequences(structure: Path, *, warmup: int = 20, measured: int = 500):
    atoms = load_atoms(structure)

    if len(atoms) != 1024 or not bool(np.all(atoms.pbc)):
        raise RuntimeError("public benchmark requires the fixed periodic 1024-atom structure")
    base = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pairs = []
    all_hashes: set[str] = set()
    for number, seed in enumerate(PUBLIC_SEEDS, start=1):
        rng = np.random.default_rng(seed)
        frames, hashes = [], []
        for index in range(warmup + measured):
            scale = 2.0e-5 + 8.0e-5 * ((index + 1) / (warmup + measured))
            positions = np.asarray(base + rng.normal(0.0, scale, base.shape), dtype=np.float64)
            digest = frame_hash(positions, cell)
            if digest in all_hashes:
                raise RuntimeError("public input sequence collision")
            all_hashes.add(digest)
            frames.append(positions)
            hashes.append(digest)
        pairs.append({"pair_id": f"Pair{number}", "seed": seed,
                      "frames": frames, "frame_hashes": hashes, "cell": cell.copy()})
    return pairs


def launch_route(*, python: Path, worker: Path, pair: dict[str, object], route: str,
                 output: Path, log: Path, model: Path, structure: Path,
                 warmup: int, measured: int, env: dict[str, str]) -> dict[str, object]:
    parent, child = socket.socketpair()
    command = [str(python), "-s", "-u", str(worker), "--route", route,
               "--model", str(model), "--structure", str(structure),
               "--output", str(output), "--warmup", str(warmup),
               "--measure", str(measured), "--protocol-fd", str(child.fileno())]
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        with log.open("wb") as log_stream:
            proc = subprocess.Popen(command, cwd=output.parent, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log_stream,
                                    stderr=log_stream, pass_fds=(child.fileno(),))
            child.close()
            protocol = parent.makefile("rwb", buffering=0)
            ready = receive(protocol)
            if not isinstance(ready, dict) or not ready.get("ready"):
                raise RouteFailure(route, f"{route} worker did not become ready: {ready}")
            for index, positions in enumerate(pair["frames"]):
                send(protocol, {"kind": "frame", "index": index, "warmup": index < warmup,
                                "positions": positions, "cell": pair["cell"]})
                reply = receive(protocol)
                if not isinstance(reply, dict) or not reply.get("ok") or reply.get("index") != index:
                    raise RouteFailure(route, f"{route} frame {index} failed: {reply}")
            send(protocol, {"kind": "finish"})
            finished = receive(protocol)
            protocol.close()
            proc.wait(timeout=300)
        if proc.returncode or not isinstance(finished, dict) or finished.get("status") != "PASS":
            raise RouteFailure(route, f"{route} worker failed rc={proc.returncode}: {finished}")
        summary = finished["summary"]
        summary["command"] = command
        write_json(Path(str(output) + ".summary.json"), summary)
        return summary
    except RouteFailure:
        raise
    except Exception as exc:
        raise RouteFailure(route, f"{route} route failed: {exc}") from exc
    finally:
        parent.close()
        child.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


def _per_frame_max(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim <= 1:
        return np.abs(value)
    return np.max(np.abs(value), axis=tuple(range(1, value.ndim)))


def validate_output(actual_path: Path, reference_path: Path, output: Path,
                    route: str, pair_id: str, tolerances: dict[str, float]) -> dict[str, object]:
    actual = np.load(actual_path)
    reference = np.load(reference_path)
    fields = {
        "energy_eV": "measured_energy_eV", "forces_eV_per_A": "measured_forces",
        "virial_eV": "measured_virial", "stress_eV_per_A3": "measured_stress",
    }
    checks: dict[str, object] = {}
    passed = True
    for tolerance_name, key in fields.items():
        if key not in actual.files or key not in reference.files:
            raise RouteFailure(route, f"{route} missing result field {key}")
        a, b = actual[key], reference[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RouteFailure(route, f"{route} {key} shape/dtype mismatch")
        errors = _per_frame_max(a.astype(np.float64) - b.astype(np.float64))
        finite = bool(np.isfinite(a).all())
        atol = float(tolerances[tolerance_name])
        field_pass = finite and bool(np.all(errors <= atol))
        passed &= field_pass
        checks[key] = {"shape": list(a.shape), "dtype": str(a.dtype), "finite": finite,
                       "max_abs": float(np.max(errors)),
                       "per_frame_max_abs": errors.tolist(), "atol": atol,
                       "status": "PASS" if field_pass else "FAIL"}
    status = "PASS" if passed else (
        "BENCHMARK_INVALID" if route == "baseline" else "CANDIDATE_INVALID")
    result = {"schema_version": "dpa4c-ppu-contest.correctness.v1",
              "pair_id": pair_id, "route": route, "status": status,
              "rtol": 0.0, "checks": checks,
              "actual_sha256": sha256(actual_path),
              "reference_sha256": sha256(reference_path)}
    write_json(output, result)
    return result


def write_latency_csv(path: Path, result_path: Path, pair_id: str, route: str, order: str) -> None:
    arrays = np.load(result_path)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "order", "route", "phase", "index", "latency_s"])
        for phase, key in (("warmup", "warmup_latencies_s"),
                           ("measured", "measured_latencies_s")):
            for index, value in enumerate(arrays[key]):
                writer.writerow([pair_id, order, route, phase, index, f"{float(value):.17g}"])


def run_benchmark(args) -> dict[str, object]:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "BENCHMARK_STATUS.json"
    status = {"schema_version": PROTOCOL, "status": "RUNNING",
              "score_type": "public_self_test", "verified": False,
              "pair_order": ["AB", "BA", "AB"], "pair_count": 3,
              "warmup": args.warmup, "measured": args.measured,
              "formal_performance": "NOT_RUN_BY_SCOPE"}
    write_json(status_path, status)
    config = json.loads(args.config.read_text())
    tolerances = config["tolerances"]
    if float(tolerances["rtol"]) != 0.0:
        raise RuntimeError("public benchmark requires rtol=0")
    pairs = generate_sequences(args.structure, warmup=args.warmup, measured=args.measured)
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
                 "CONDA_DEFAULT_ENV", "PYTHONUSERBASE", "LD_PRELOAD"):
        env.pop(name, None)
    env.update({"PYTHONNOUSERSITE": "1", "DP_COMPILE_INFER": "0",
                "DP_TF32_INFER": "0", "DP_AMP_INFER": "0"})
    records, pair_records = [], []
    overall = "PASS"
    for pair, routes in zip(pairs, PAIR_ORDERS, strict=True):
        pair_id = str(pair["pair_id"])
        pair_dir = root / pair_id
        pair_dir.mkdir()
        reference_path = pair_dir / "reference.npz"
        launch_route(python=args.baseline_python, worker=args.worker, pair=pair,
                     route="reference", output=reference_path,
                     log=pair_dir / "reference.log", model=args.model,
                     structure=args.structure, warmup=args.warmup,
                     measured=args.measured, env=env)
        reference_manifest = {"schema_version": "dpa4c-ppu-contest.reference.v1",
                              "pair_id": pair_id, "result_sha256": sha256(reference_path),
                              "summary_sha256": sha256(Path(str(reference_path) + ".summary.json"))}
        reference_manifest_path = pair_dir / "reference-manifest.json"
        write_json(reference_manifest_path, reference_manifest)
        pair_result: dict[str, object] = {
            "pair_id": pair_id,
            "order": "".join("A" if route == "baseline" else "B" for route in routes),
            "reference_manifest_sha256": sha256(reference_manifest_path), "routes": {},
        }
        for route in routes:
            route_path = pair_dir / f"{route}.npz"
            python = args.candidate_python if route == "candidate" else args.baseline_python
            summary = launch_route(python=python, worker=args.worker, pair=pair, route=route,
                                   output=route_path, log=pair_dir / f"{route}.log",
                                   model=args.model, structure=args.structure,
                                   warmup=args.warmup, measured=args.measured, env=env)
            correctness = validate_output(route_path, reference_path,
                                          pair_dir / f"{route}.correctness.json",
                                          route, pair_id, tolerances)
            write_latency_csv(pair_dir / f"{route}.latency.csv", route_path, pair_id,
                              route, pair_result["order"])
            record = {"pair_id": pair_id, "order": pair_result["order"], "route": route,
                      "pid": summary["pid"], "correctness": correctness,
                      "latency": summary["latency"], "result_npz": str(route_path),
                      "result_npz_sha256": sha256(route_path)}
            records.append(record)
            pair_result["routes"][route] = record
        baseline = pair_result["routes"]["baseline"]
        candidate = pair_result["routes"]["candidate"]
        if baseline["correctness"]["status"] != "PASS":
            overall = "BENCHMARK_INVALID"
        elif candidate["correctness"]["status"] != "PASS":
            overall = "CANDIDATE_INVALID"
        input_path = pair_dir / "inputs.npz"
        np.savez_compressed(input_path, positions=np.stack(pair["frames"]), cell=pair["cell"])
        hashes = pair["frame_hashes"]
        input_manifest = {"schema_version": "dpa4c-ppu-contest.public-input.v1",
                          "pair_id": pair_id, "public_seed": pair["seed"],
                          "frame_count": len(hashes), "warmup_count": args.warmup,
                          "measured_count": args.measured,
                          "warmup_frame_hashes": hashes[:args.warmup],
                          "measured_frame_hashes": hashes[args.warmup:],
                          "within_pair_disjoint": not bool(set(hashes[:args.warmup]) & set(hashes[args.warmup:])),
                          "across_public_pairs_disjoint": True,
                          "inputs_npz_sha256": sha256(input_path)}
        input_manifest_path = pair_dir / "input-manifest.json"
        write_json(input_manifest_path, input_manifest)
        pair_result["input_manifest_sha256"] = sha256(input_manifest_path)
        pair_result["speedup_candidate_over_baseline"] = (
            baseline["latency"]["mean_s"] / candidate["latency"]["mean_s"])
        pair_records.append(pair_result)
        write_json(pair_dir / "result.json", pair_result)
        if overall != "PASS":
            break
    speedups = [float(pair["speedup_candidate_over_baseline"]) for pair in pair_records]
    final_status = overall if overall != "PASS" else ("PASS" if len(pair_records) == 3 else "BENCHMARK_INVALID")
    aggregate = {"schema_version": "dpa4c-ppu-contest.public-aggregate.v1",
                 "status": final_status, "score_type": "public_self_test", "verified": False,
                 "paired_median_speedup": (
                     float(statistics.median(speedups)) if final_status == "PASS" else None
                 ),
                 "pair_speedups_candidate_over_baseline": speedups, "pairs": pair_records}
    write_json(root / "formal-aggregate.json", aggregate)
    write_json(root / "repeats.json", {"schema_version": PROTOCOL, "routes": records})
    write_json(root / "result.json", aggregate)
    with (root / "formal-repeat.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "order", "route", "pid", "correctness", "mean_s", "p50_s", "p90_s", "p99_s", "cv", "evals_per_s", "atoms_per_s", "result_npz_sha256"])
        for record in records:
            latency = record["latency"]
            writer.writerow([record["pair_id"], record["order"], record["route"], record["pid"], record["correctness"]["status"], latency["mean_s"], latency["p50_s"], latency["p90_s"], latency["p99_s"], latency["cv"], latency["throughput_evals_per_s"], latency["throughput_atoms_per_s"], record["result_npz_sha256"]])
    with (root / "formal-aggregate.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "order", "baseline_evals_per_s", "candidate_evals_per_s", "speedup_candidate_over_baseline", "baseline_correctness", "candidate_correctness"])
        for pair in pair_records:
            baseline, candidate = pair["routes"]["baseline"], pair["routes"]["candidate"]
            writer.writerow([pair["pair_id"], pair["order"], baseline["latency"]["throughput_evals_per_s"], candidate["latency"]["throughput_evals_per_s"], pair["speedup_candidate_over_baseline"], baseline["correctness"]["status"], candidate["correctness"]["status"]])
    status.update({"status": final_status, "paired_median_speedup": aggregate["paired_median_speedup"]})
    write_json(status_path, status)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=500)
    args = parser.parse_args()
    try:
        result = run_benchmark(args)
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        status = getattr(exc, "status", "BENCHMARK_INVALID")
        write_json(args.output_root / "BENCHMARK_STATUS.json",
                   {"schema_version": PROTOCOL, "status": status,
                    "score_type": "public_self_test", "verified": False,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "formal_performance": "NOT_RUN_BY_SCOPE"})
        print(json.dumps({"status": status, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
