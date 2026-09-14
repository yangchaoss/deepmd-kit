#!/usr/bin/env python3
"""Compare isolated baseline and candidate public-smoke outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["tolerances"]
    limits = {
        "energy_eV": float(config["energy_eV"]),
        "forces_eV_per_A": float(config["forces_eV_per_A"]),
        "virial_eV": float(config["virial_eV"]),
        "stress_eV_per_A3": float(config["stress_eV_per_A3"]),
    }
    if float(config["rtol"]) != 0.0:
        raise SystemExit("public smoke requires rtol=0")
    baseline = np.load(args.baseline)
    candidate = np.load(args.candidate)
    maxima: dict[str, float] = {name: 0.0 for name in limits}
    checks: dict[str, object] = {}
    for key in sorted(baseline.files):
        if key not in candidate.files:
            raise SystemExit(f"candidate output missing {key}")
        field = key.split("__", 1)[1]
        a, b = baseline[key], candidate[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise SystemExit(f"shape/dtype mismatch for {key}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        error = float(np.max(np.abs(a - b))) if a.shape else float(abs(a - b))
        maxima[field] = max(maxima[field], error)
        checks[key] = {"max_abs": error, "atol": limits[field], "status": "PASS" if error <= limits[field] else "FAIL"}
    status = "PASS" if all(value <= limits[name] for name, value in maxima.items()) else "FAIL"
    result = {
        "schema_version": "dpa4c-ppu-contest.public-smoke.v1",
        "status": status,
        "rtol": 0.0,
        "max_abs": maxima,
        "limits": limits,
        "checks": checks,
        "formal_performance": "NOT_RUN_BY_SCOPE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
