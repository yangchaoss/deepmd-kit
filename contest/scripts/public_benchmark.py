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


PROTOCOL = "dpa4c-ppu-contest.public-benchmark.v2"
PUBLIC_SEEDS = (510421, 620531, 730637)
PAIR_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"))
OUTPUT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "config" / "output-contract.json"


def aggregation_method(profile: str) -> str:
    if profile == "quick":
        return "paired_speedup"
    if profile == "full":
        return "paired_geometric_mean_speedup"
    raise ValueError(f"unsupported profile: {profile}")


def aggregate_speedups(speedups: list[float], method: str) -> float | None:
    if not speedups:
        return None
    if method == "paired_speedup":
        if len(speedups) != 1:
            raise ValueError("paired_speedup requires exactly one pair")
        return float(speedups[0])
    if method == "paired_geometric_mean_speedup":
        if len(speedups) != 2 or any(value <= 0.0 for value in speedups):
            raise ValueError("paired_geometric_mean_speedup requires two positive pairs")
        return float(np.sqrt(speedups[0] * speedups[1]))
    raise ValueError(f"unsupported aggregation method: {method}")


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


COMPACT_FIELDS = {
    "energy_eV": "measured_energy_eV",
    "forces_eV_per_A": "measured_forces",
    "virial_eV": "measured_virial",
    "stress_eV_per_A3": "measured_stress",
}
TIMING_FIELDS = ("warmup_latencies_s", "measured_latencies_s")
ARCHIVE_FIELDS = set(COMPACT_FIELDS.values()) | set(TIMING_FIELDS)


def load_output_contract(path: Path = OUTPUT_CONTRACT_PATH) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "dpa4c-ppu-contest.output-contract.v1":
        raise RuntimeError("unsupported output contract schema")
    fields = config.get("fields")
    if not isinstance(fields, dict) or set(fields) != ARCHIVE_FIELDS:
        raise RuntimeError("output contract fields are incomplete")
    if (config.get("measured_dimension") != "N"
            or config.get("warmup_dimension") != "W"
            or int(config.get("atom_count", -1)) != 1024):
        raise RuntimeError("output contract dimensions are invalid")
    for name, spec in fields.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("shape"), list):
            raise RuntimeError(f"output contract shape is invalid for {name}")
        if not spec.get("allowed_dtypes"):
            raise RuntimeError(f"output contract dtype is missing for {name}")
        if spec.get("category") not in {"physics", "timing"}:
            raise RuntimeError(f"output contract category is invalid for {name}")
        if name in TIMING_FIELDS and (spec["category"] != "timing" or spec.get("positive") is not True):
            raise RuntimeError(f"output contract timing rule is invalid for {name}")
        if name in COMPACT_FIELDS.values() and spec["category"] != "physics":
            raise RuntimeError(f"output contract physics rule is invalid for {name}")
    return config


def output_contract_identity(path: Path = OUTPUT_CONTRACT_PATH) -> dict[str, object]:
    config = load_output_contract(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema_version": config["schema_version"],
        "measured_dimension": config["measured_dimension"],
        "warmup_dimension": config["warmup_dimension"],
        "atom_count": config["atom_count"],
        "fields": config["fields"],
    }


def _compact_correctness_summary(pairs: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate correctness records without carrying per-frame arrays forward."""
    routes = {"baseline": [], "candidate": []}
    for pair in pairs:
        for route in routes:
            record = pair.get("routes", {}).get(route)
            if not isinstance(record, dict):
                raise ValueError(f"missing {route} correctness record in {pair.get('pair_id')}")
            routes[route].append(record)

    summary_routes: dict[str, object] = {}
    tolerance_summary: dict[str, dict[str, float]] = {}
    route_statuses: dict[str, str] = {}
    for route, records in routes.items():
        shapes: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        finite: dict[str, bool] = {}
        max_abs: dict[str, float] = {}
        measured_frames: int | None = None
        route_status = "PASS"
        for record in records:
            correctness = record.get("correctness", {})
            status = str(correctness.get("status", "FAIL"))
            if status != "PASS":
                route_status = status
            checks = correctness.get("checks", {})
            if not isinstance(checks, dict):
                raise ValueError(f"missing correctness checks in {route} {record.get('pair_id')}")
            for field, check_name in COMPACT_FIELDS.items():
                check = checks.get(check_name)
                if not isinstance(check, dict):
                    raise ValueError(f"missing {check_name} correctness check")
                shape = list(check.get("shape", []))
                dtype = str(check.get("dtype", ""))
                if not shape or not dtype:
                    raise ValueError(f"incomplete {check_name} correctness check")
                if field not in shapes:
                    shapes[field] = shape
                    dtypes[field] = dtype
                    finite[field] = bool(check.get("finite", check.get("isfinite", False)))
                    max_abs[field] = float(check["max_abs"])
                else:
                    if shape != shapes[field] or dtype != dtypes[field]:
                        raise ValueError(f"inconsistent {field} shape/dtype across pairs")
                    finite[field] = finite[field] and bool(
                        check.get("finite", check.get("isfinite", False))
                    )
                    max_abs[field] = max(max_abs[field], float(check["max_abs"]))
                frames = int(shape[0])
                if measured_frames is None:
                    measured_frames = frames
                elif measured_frames != frames:
                    raise ValueError("inconsistent measured frame count across fields")
                atol, rtol = float(check["atol"]), float(check["rtol"])
                prior = tolerance_summary.get(field)
                value = {"atol": atol, "rtol": rtol}
                if prior is not None and prior != value:
                    raise ValueError(f"inconsistent tolerance for {field}")
                tolerance_summary[field] = value
        summary_routes[route] = {
            "measured_frames": measured_frames,
            "shape": shapes,
            "dtype": dtypes,
            "isfinite": finite,
            "max_abs": max_abs,
        }
        route_statuses[route] = route_status

    if route_statuses["baseline"] != "PASS":
        overall = "BENCHMARK_INVALID"
    elif route_statuses["candidate"] != "PASS":
        overall = "CANDIDATE_INVALID"
    else:
        overall = "PASS"
    return {
        "status": overall,
        "routes": summary_routes,
        "tolerance": tolerance_summary,
    }


def compact_public_result(
    aggregate: dict[str, object], *, warmup: int = 20, measured: int = 100
) -> dict[str, object]:
    """Return the compact public result while leaving full evidence in repeats.json."""
    pairs = aggregate.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("aggregate pairs are missing")
    if not pairs:
        raise ValueError("aggregate pairs are empty")
    correctness = _compact_correctness_summary(pairs)
    pair_speedups = [float(pair["speedup_candidate_over_baseline"]) for pair in pairs]
    compact_pairs = []
    latencies = {"baseline": [], "candidate": []}
    for pair in pairs:
        compact_routes = {}
        for route in ("baseline", "candidate"):
            record = pair["routes"][route]
            latency = record["latency"]
            latencies[route].append(latency)
            compact_routes[route] = {
                "latency": {
                    key: float(latency[key])
                    for key in ("mean_s", "p50_s", "p90_s", "p99_s", "cv")
                },
                "throughput": {
                    "evals_per_s": float(latency["throughput_evals_per_s"]),
                    "atoms_per_s": float(latency["throughput_atoms_per_s"]),
                },
                "correctness": record["correctness"]["status"],
            }
        compact_pairs.append({
            "pair_id": pair["pair_id"],
            "order": pair["order"],
            **compact_routes,
            "speedup_candidate_over_baseline": float(
                pair["speedup_candidate_over_baseline"]
            ),
        })

    def median(route: str, key: str) -> float:
        return float(statistics.median(float(item[key]) for item in latencies[route]))

    method = str(aggregate.get("aggregation_method", aggregation_method(str(aggregate.get("profile", "full")))))
    aggregate_speedup = aggregate_speedups(pair_speedups, method)
    performance = {
        "fresh_pair_count": len(pairs),
        "fresh_process_count": len(pairs) * 2,
        "baseline": {
            "median_evals_per_s": median("baseline", "throughput_evals_per_s"),
            "median_atoms_per_s": median("baseline", "throughput_atoms_per_s"),
            "runtime_cv_median": median("baseline", "cv"),
        },
        "candidate": {
            "median_evals_per_s": median("candidate", "throughput_evals_per_s"),
            "median_atoms_per_s": median("candidate", "throughput_atoms_per_s"),
            "runtime_cv_median": median("candidate", "cv"),
        },
        "candidate_latency_s": {
            "p50_median": median("candidate", "p50_s"),
            "p90_median": median("candidate", "p90_s"),
            "p99_median": median("candidate", "p99_s"),
        },
        "pair_speedups_candidate_over_baseline": pair_speedups,
        "aggregation_method": method,
    }
    if method == "paired_geometric_mean_speedup":
        performance["paired_geometric_mean_speedup"] = aggregate_speedup
    else:
        performance["paired_speedup"] = aggregate_speedup
    status = str(aggregate.get("status", "BENCHMARK_INVALID"))
    if status == "PASS" and correctness["status"] != "PASS":
        status = str(correctness["status"])
    return {
        "schema_version": aggregate["schema_version"],
        "result_format": "compact",
        "status": status,
        "profile": aggregate.get("profile", "full"),
        "score_type": aggregate.get("score_type", "public_self_test"),
        "verified": bool(aggregate.get("verified", False)),
        "formal_performance": "NOT_RUN_BY_SCOPE",
        "protocol": {
            "pair_order": [str(pair["order"]) for pair in pairs],
            "pair_count": len(pairs),
            "warmup": warmup,
            "measured": measured,
            "aggregation_method": method,
        },
        "aggregation_method": method,
        "benchmark_tolerance": aggregate.get("benchmark_tolerance"),
        "output_contract": aggregate.get("output_contract"),
        "pair_speedups_candidate_over_baseline": pair_speedups,
        ("paired_geometric_mean_speedup" if method == "paired_geometric_mean_speedup"
         else "paired_speedup"): aggregate_speedup if status == "PASS" else None,
        "pairs": compact_pairs,
        "correctness_summary": correctness,
        "performance_summary": performance,
    }


def load_tolerance(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "dpa4c-ppu-contest.benchmark-tolerance.v1":
        raise RuntimeError("unsupported benchmark tolerance schema")
    fields = config.get("fields", {})
    expected = {"energy_eV", "forces_eV_per_A", "virial_eV", "stress_eV_per_A3"}
    if set(fields) != expected:
        raise RuntimeError("benchmark tolerance fields are incomplete")
    for name, values in fields.items():
        if float(values["atol"]) < 0.0 or float(values["rtol"]) < 0.0:
            raise RuntimeError(f"negative benchmark tolerance for {name}")
    return config


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


def generate_sequences(
    structure: Path, *, warmup: int = 20, measured: int = 100, pair_count: int = 2
):
    if pair_count not in (1, 2):
        raise RuntimeError("pair_count must be 1 (quick) or 2 (full); legacy 3-pair protocol is unsupported")
    atoms = load_atoms(structure)

    if len(atoms) != 1024 or not bool(np.all(atoms.pbc)):
        raise RuntimeError("public benchmark requires the fixed periodic 1024-atom structure")
    base = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pairs = []
    all_hashes: set[str] = set()
    for number, seed in enumerate(PUBLIC_SEEDS[:pair_count], start=1):
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
                    route: str, pair_id: str, tolerance: dict[str, object],
                    tolerance_path: Path,
                    output_contract: dict[str, object] | None = None,
                    *, warmup: int | None = None,
                    measured: int | None = None) -> dict[str, object]:
    actual = np.load(actual_path)
    reference = np.load(reference_path)
    output_contract = output_contract or load_output_contract()
    fields = COMPACT_FIELDS

    def validate_contract(data, role: str) -> dict[str, object]:
        expected_fields = set(output_contract["fields"])
        actual_fields = set(data.files)
        if actual_fields != expected_fields:
            raise RouteFailure(role, f"{role} output field set mismatch: {sorted(actual_fields)}")
        measured_frames = None
        info = {}

        def check_field(field_name: str, value: np.ndarray, expected_shape: tuple[int, ...]) -> None:
            spec = output_contract["fields"][field_name]
            if value.ndim == 0:
                raise RouteFailure(role, f"{role} {field_name} is scalar")
            if tuple(value.shape) != expected_shape:
                raise RouteFailure(role, f"{role} {field_name} shape {list(value.shape)} != {list(expected_shape)}")
            if str(value.dtype) not in set(spec["allowed_dtypes"]):
                raise RouteFailure(role, f"{role} {field_name} dtype {value.dtype} is not allowed")
            if not bool(np.isfinite(value).all()):
                raise RouteFailure(role, f"{role} {field_name} contains non-finite values")
            if spec.get("positive") is True and not bool(np.all(value > 0)):
                raise RouteFailure(role, f"{role} {field_name} must be finite and strictly positive")
            info[field_name] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": True}

        for field_name in sorted(COMPACT_FIELDS.values()):
            value = data[field_name]
            if value.ndim == 0:
                raise RouteFailure(role, f"{role} {field_name} is scalar")
            if measured_frames is None:
                measured_frames = int(value.shape[0])
            elif int(value.shape[0]) != measured_frames:
                raise RouteFailure(role, f"{role} measured frame dimension is inconsistent")
        if measured is not None and measured_frames != int(measured):
            raise RouteFailure(role, f"{role} measured frame dimension {measured_frames} != protocol {measured}")
        if measured_frames is None or measured_frames <= 0:
            raise RouteFailure(role, f"{role} measured frame dimension is invalid")

        for field_name in sorted(COMPACT_FIELDS.values()):
            value = data[field_name]
            spec = output_contract["fields"][field_name]
            expected_shape = tuple(
                measured_frames if dimension == "N" else int(dimension)
                for dimension in spec["shape"]
            )
            check_field(field_name, value, expected_shape)

        warmup_frames = int(warmup) if warmup is not None else None
        if warmup_frames is not None and warmup_frames <= 0:
            raise RouteFailure(role, f"{role} warmup frame dimension is invalid")
        for field_name in TIMING_FIELDS:
            value = data[field_name]
            if value.ndim == 0:
                raise RouteFailure(role, f"{role} {field_name} is scalar")
            if warmup_frames is None and field_name == "warmup_latencies_s":
                warmup_frames = int(value.shape[0])
            expected_count = warmup_frames if field_name == "warmup_latencies_s" else measured_frames
            check_field(field_name, value, (expected_count,))
        return info

    reference_contract = validate_contract(reference, "reference")
    actual_contract = validate_contract(actual, route)
    for field_name in TIMING_FIELDS:
        if actual[field_name].shape != reference[field_name].shape:
            raise RouteFailure(route, f"{route} {field_name} shape mismatch")
        if actual[field_name].dtype != reference[field_name].dtype:
            raise RouteFailure(route, f"{route} {field_name} dtype mismatch")
    checks: dict[str, object] = {}
    passed = True
    for tolerance_name, key in fields.items():
        a, b = actual[key], reference[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RouteFailure(route, f"{route} {key} shape/dtype mismatch")
        errors = _per_frame_max(a.astype(np.float64) - b.astype(np.float64))
        finite = bool(np.isfinite(a).all())
        values = tolerance["fields"][tolerance_name]
        atol, rtol = float(values["atol"]), float(values["rtol"])
        allowed = atol + rtol * np.abs(b.astype(np.float64))
        field_pass = finite and bool(np.all(np.abs(a.astype(np.float64) - b.astype(np.float64)) <= allowed))
        passed &= field_pass
        checks[key] = {"shape": list(a.shape), "dtype": str(a.dtype), "finite": finite,
                       "max_abs": float(np.max(errors)),
                       "per_frame_max_abs": errors.tolist(), "atol": atol, "rtol": rtol,
                       "status": "PASS" if field_pass else "FAIL"}
    status = "PASS" if passed else (
        "BENCHMARK_INVALID" if route == "baseline" else "CANDIDATE_INVALID")
    result = {"schema_version": "dpa4c-ppu-contest.correctness.v1",
              "pair_id": pair_id, "route": route, "status": status,
              "output_contract": output_contract_identity(
                  OUTPUT_CONTRACT_PATH
              ),
              "contract": {"reference": reference_contract, "actual": actual_contract},
              "tolerance": {"path": str(tolerance_path.resolve()),
                            "sha256": sha256(tolerance_path),
                            "tolerance_id": tolerance["tolerance_id"],
                            "fields": tolerance["fields"]},
              "checks": checks,
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
    score_type = "development_quick" if args.profile == "quick" else "public_self_test"
    method = aggregation_method(args.profile)
    pair_orders = PAIR_ORDERS[:args.pair_count]
    status = {"schema_version": PROTOCOL, "status": "RUNNING",
              "profile": args.profile, "score_type": score_type, "verified": False,
              "pair_order": ["AB" if order == ("baseline", "candidate") else "BA"
                              for order in pair_orders],
              "pair_count": args.pair_count,
              "warmup": args.warmup, "measured": args.measured,
              "aggregation_method": method,
              "formal_performance": "NOT_RUN_BY_SCOPE"}
    write_json(status_path, status)
    tolerance = load_tolerance(args.benchmark_tolerance)
    output_contract = load_output_contract()
    output_contract_id = output_contract_identity()
    tolerance_identity = {"path": str(args.benchmark_tolerance.resolve()),
                          "sha256": sha256(args.benchmark_tolerance),
                          "tolerance_id": tolerance["tolerance_id"],
                          "fields": tolerance["fields"]}
    status["benchmark_tolerance"] = tolerance_identity
    write_json(status_path, status)
    pairs = generate_sequences(
        args.structure, warmup=args.warmup, measured=args.measured,
        pair_count=args.pair_count,
    )
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
                 "CONDA_DEFAULT_ENV", "PYTHONUSERBASE", "LD_PRELOAD"):
        env.pop(name, None)
    env.update({"PYTHONNOUSERSITE": "1", "DP_COMPILE_INFER": "0",
                "DP_TF32_INFER": "0", "DP_AMP_INFER": "0"})
    records, pair_records = [], []
    overall = "PASS"
    for pair, routes in zip(pairs, pair_orders, strict=True):
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
                                          route, pair_id, tolerance, args.benchmark_tolerance,
                                          output_contract, warmup=args.warmup,
                                          measured=args.measured)
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
    final_status = overall if overall != "PASS" else (
        "PASS" if len(pair_records) == args.pair_count else "BENCHMARK_INVALID"
    )
    aggregate_speedup = aggregate_speedups(speedups, method) if final_status == "PASS" else None
    aggregate = {"schema_version": "dpa4c-ppu-contest.public-aggregate.v2",
                 "status": final_status, "profile": args.profile,
                 "score_type": score_type, "verified": False,
                 "pair_order": ["AB" if order == ("baseline", "candidate") else "BA"
                                for order in pair_orders],
                 "pair_count": args.pair_count,
                 "warmup": args.warmup,
                 "measured": args.measured,
                 "aggregation_method": method,
                 "benchmark_tolerance": tolerance_identity,
                 "output_contract": output_contract_id,
                 "pair_speedups_candidate_over_baseline": speedups, "pairs": pair_records}
    aggregate["paired_geometric_mean_speedup" if method == "paired_geometric_mean_speedup"
              else "paired_speedup"] = aggregate_speedup
    write_json(root / "formal-aggregate.json", aggregate)
    write_json(root / "repeats.json", {
        "schema_version": PROTOCOL,
        "profile": args.profile,
        "score_type": score_type,
        "verified": False,
        "pair_order": ["AB" if order == ("baseline", "candidate") else "BA"
                        for order in pair_orders],
        "pair_count": args.pair_count,
        "warmup": args.warmup,
        "measured": args.measured,
        "aggregation_method": method,
        "output_contract": output_contract_id,
        "routes": records,
    })
    write_json(root / "result.json", compact_public_result(
        aggregate, warmup=args.warmup, measured=args.measured
    ))
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
    status.update({"status": final_status,
                   ("paired_geometric_mean_speedup" if method == "paired_geometric_mean_speedup"
                    else "paired_speedup"): aggregate_speedup})
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
    parser.add_argument("--benchmark-tolerance", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--pair-count", type=int, default=2)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    args = parser.parse_args()
    try:
        result = run_benchmark(args)
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        status = getattr(exc, "status", "BENCHMARK_INVALID")
        score_type = "development_quick" if args.profile == "quick" else "public_self_test"
        write_json(args.output_root / "BENCHMARK_STATUS.json",
                   {"schema_version": PROTOCOL, "status": status,
                    "profile": args.profile, "score_type": score_type,
                    "verified": False, "pair_count": args.pair_count,
                    "pair_order": ["AB", "BA"][:args.pair_count],
                    "warmup": args.warmup, "measured": args.measured,
                    "aggregation_method": aggregation_method(args.profile),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "formal_performance": "NOT_RUN_BY_SCOPE"})
        print(json.dumps({"status": status, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
