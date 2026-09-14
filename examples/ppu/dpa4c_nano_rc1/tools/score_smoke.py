#!/usr/bin/env python3
"""Validate a public smoke result and emit a non-formal score preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "energy_max_abs": 3.4e-4,
    "forces_max_abs": 5.5e-5,
    "virial_max_abs": 5.0e-4,
    "stress_max_abs": 1.0e-7,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    errors: list[str] = []
    if smoke.get("schema_version") != "dpa4c-contest-runtime.efs-demo.v1":
        errors.append("unexpected smoke schema")
    if smoke.get("status") != "PASS":
        errors.append("smoke status is not PASS")
    for case_name in ("base", "microperturbation"):
        case = smoke.get("cases", {}).get(case_name, {})
        diff = case.get("diff", {})
        for field, limit in EXPECTED.items():
            value = float(diff.get(field, float("inf")))
            if value > limit:
                errors.append(f"{case_name}/{field}={value} exceeds {limit}")
        for route in ("baseline", "candidate"):
            output = case.get(route, {})
            expected_shapes = {"energy_eV": [], "forces_eV_per_A": [1024, 3], "virial_eV": [3, 3], "stress_eV_per_A3": [6]}
            for quantity, shape in expected_shapes.items():
                item = output.get(quantity, {})
                if item.get("shape") != shape or item.get("dtype") != "float64" or item.get("isfinite") is not True:
                    errors.append(f"{case_name}/{route}/{quantity} shape-dtype-finite mismatch")
    timing = smoke.get("timing", {})
    if timing.get("warmup") != 1 or timing.get("measure") != 2:
        errors.append("public smoke must use 1 warmup + 2 measured")
    speedup = float(timing.get("preview_speedup", 0.0))
    if not speedup > 0.0:
        errors.append("preview speedup is not positive")
    result = {
        "schema_version": "dpa4c-contest-runtime.public-score-preview.v1",
        "status": "PASS" if not errors else "FAIL",
        "scope": "PUBLIC_SMOKE_ONLY",
        "formal_score": "NOT_COMPUTED",
        "preview_metric": {"name": "baseline_p50_over_candidate_p50", "value": speedup},
        "protocol": {"model": "Nano", "atoms": 1024, "precision": "FP32", "outer": "eager", "warmup": 1, "measured": 2},
        "smoke_result": {"path": str(args.smoke.resolve()), "sha256": sha256(args.smoke)},
        "errors": errors,
        "policy": "The organizer rebuilds the candidate and applies the private paired 20+500 protocol for any final score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
