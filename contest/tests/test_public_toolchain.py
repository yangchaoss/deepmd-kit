from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


flow = load("flow")
benchmark = load("public_benchmark")


class FakeCell:
    array = np.eye(3, dtype=np.float64) * 20.0


class FakeAtoms:
    positions = np.arange(3072, dtype=np.float64).reshape(1024, 3) / 100.0
    pbc = np.ones(3, dtype=bool)
    cell = FakeCell()

    def __len__(self):
        return 1024


def valid_output_fields(frames=2, dtype=np.float64, warmup=2):
    return {
        "measured_energy_eV": np.zeros((frames,), dtype=dtype),
        "measured_forces": np.zeros((frames, 1024, 3), dtype=dtype),
        "measured_virial": np.zeros((frames, 3, 3), dtype=dtype),
        "measured_stress": np.zeros((frames, 6), dtype=dtype),
        "warmup_latencies_s": np.ones((warmup,), dtype=dtype),
        "measured_latencies_s": np.ones((frames,), dtype=dtype),
    }


def synthetic_aggregate() -> dict[str, object]:
    fields = {
        "measured_energy_eV": ([4], "float64", 0.001),
        "measured_forces": ([4, 1024, 3], "float64", 0.002),
        "measured_virial": ([4, 3, 3], "float64", 0.003),
        "measured_stress": ([4, 6], "float64", 0.004),
    }
    pairs = []
    for number, order in enumerate(("AB", "BA"), start=1):
        routes = {}
        for route in ("baseline", "candidate"):
            checks = {
                name: {
                    "shape": shape, "dtype": dtype, "finite": True,
                    "max_abs": maximum + number * 0.001 + (0.001 if route == "candidate" else 0.0),
                    "per_frame_max_abs": [maximum, maximum + 0.001],
                    "atol": 0.01, "rtol": 0.0, "status": "PASS",
                }
                for name, (shape, dtype, maximum) in fields.items()
            }
            routes[route] = {
                "correctness": {"status": "PASS", "checks": checks},
                "latency": {
                    "mean_s": 1.0 + number * 0.1 + (0.01 if route == "candidate" else 0.0),
                    "p50_s": 0.9 + number * 0.1 + (0.01 if route == "candidate" else 0.0),
                    "p90_s": 1.1 + number * 0.1 + (0.01 if route == "candidate" else 0.0),
                    "p99_s": 1.2 + number * 0.1 + (0.01 if route == "candidate" else 0.0),
                    "cv": 0.01 * number + (0.001 if route == "candidate" else 0.0),
                    "throughput_evals_per_s": 10.0 - number + (0.2 if route == "candidate" else 0.0),
                    "throughput_atoms_per_s": 100.0 - number + (2.0 if route == "candidate" else 0.0),
                },
            }
        pairs.append({
            "pair_id": f"Pair{number}", "order": order, "routes": routes,
            "speedup_candidate_over_baseline": 1.0 + number * 0.01,
        })
    return {
        "schema_version": "dpa4c-ppu-contest.public-aggregate.v2",
        "status": "PASS", "score_type": "public_self_test", "verified": False,
        "profile": "full", "aggregation_method": "paired_geometric_mean_speedup",
        "benchmark_tolerance": {"fields": {"energy_eV": {"atol": 0.01, "rtol": 0.0}}},
        "pair_speedups_candidate_over_baseline": [1.01, 1.02], "pairs": pairs,
    }


class PublicBenchmarkTest(unittest.TestCase):
    def test_compact_summary_aggregates_records_and_keeps_repeats_separate(self):
        aggregate = synthetic_aggregate()
        compact = benchmark.compact_public_result(aggregate, warmup=2, measured=4)
        self.assertEqual(compact["status"], "PASS")
        self.assertEqual(compact["result_format"], "compact")
        self.assertEqual(compact["protocol"], {
            "pair_order": ["AB", "BA"], "pair_count": 2,
            "warmup": 2, "measured": 4,
            "aggregation_method": "paired_geometric_mean_speedup",
        })
        correctness = compact["correctness_summary"]
        self.assertEqual(correctness["status"], "PASS")
        self.assertEqual(correctness["routes"]["candidate"]["measured_frames"], 4)
        self.assertEqual(correctness["routes"]["candidate"]["shape"]["forces_eV_per_A"], [4, 1024, 3])
        self.assertEqual(correctness["routes"]["candidate"]["max_abs"]["energy_eV"], 0.004)
        self.assertEqual(correctness["tolerance"]["energy_eV"], {"atol": 0.01, "rtol": 0.0})
        performance = compact["performance_summary"]
        self.assertEqual(performance["fresh_pair_count"], 2)
        self.assertEqual(performance["fresh_process_count"], 4)
        self.assertEqual(performance["candidate"]["median_evals_per_s"], 8.7)
        self.assertEqual(performance["candidate_latency_s"]["p50_median"], 1.06)
        self.assertEqual(performance["aggregation_method"], "paired_geometric_mean_speedup")
        self.assertAlmostEqual(performance["paired_geometric_mean_speedup"], (1.01 * 1.02) ** 0.5)
        self.assertAlmostEqual(compact["paired_geometric_mean_speedup"], (1.01 * 1.02) ** 0.5)
        self.assertNotIn("paired_median_speedup", json.dumps(compact))
        encoded = json.dumps(compact)
        self.assertNotIn("per_frame_max_abs", encoded)
        self.assertNotIn("warmup_latencies_s", encoded)
        repeats = {"routes": [{"correctness": {"checks": {
            "measured_energy_eV": {"per_frame_max_abs": [0.1, 0.2]}
        }}}]}
        self.assertEqual(repeats["routes"][0]["correctness"]["checks"]["measured_energy_eV"]["per_frame_max_abs"], [0.1, 0.2])

    def test_compact_summary_cannot_hide_correctness_failure(self):
        aggregate = synthetic_aggregate()
        aggregate["pairs"][1]["routes"]["candidate"]["correctness"]["status"] = "CANDIDATE_INVALID"
        compact = benchmark.compact_public_result(aggregate)
        self.assertEqual(compact["status"], "CANDIDATE_INVALID")
        self.assertEqual(compact["correctness_summary"]["status"], "CANDIDATE_INVALID")
        self.assertEqual(compact["pairs"][1]["candidate"]["correctness"], "CANDIDATE_INVALID")

    def test_full_aggregation_is_geometric_and_quick_is_single_pair(self):
        self.assertEqual(benchmark.aggregate_speedups([1.0], "paired_speedup"), 1.0)
        self.assertEqual(benchmark.aggregate_speedups([1.0, 4.0], "paired_geometric_mean_speedup"), 2.0)
        with self.assertRaisesRegex(ValueError, "requires two"):
            benchmark.aggregate_speedups([1.0, 2.0, 3.0], "paired_geometric_mean_speedup")

    def test_fixed_submission_contract_and_sha_lines(self):
        expected = {
            "candidate.patch", "result.json", "repeats.json", "measurement-binding.json",
            "submission-manifest.json", "image.json", "CHANGELOG.md", "SHA256SUMS",
        }
        self.assertEqual(flow.SUBMISSION_FILES, expected)
        self.assertEqual(len(flow.SUBMISSION_FILES), 8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in sorted(expected - {"SHA256SUMS"}):
                (root / name).write_text(name + "\n")
            payloads = sorted(root.iterdir())
            lines = "".join(f"{flow.sha256(path)}  {path.name}\n" for path in payloads)
            (root / "SHA256SUMS").write_text(lines)
            for line in (root / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, flow.sha256(root / name))

    def test_runtime_requirements_are_hash_locked(self):
        lines = flow.RUNTIME_REQUIREMENTS.read_text().splitlines()
        self.assertEqual(len(lines), 18)
        self.assertEqual(
            {line.split("==", 1)[0] for line in lines},
            {
                "array-api-compat", "ase", "bracex", "colorama", "dargs",
                "deprecated", "flexcache", "flexparser", "greenlet", "h5py",
                "lmdb", "mendeleev", "pint", "pyfiglet", "sqlalchemy",
                "typeguard", "wcmatch", "wrapt",
            },
        )
        self.assertTrue(all(" --hash=sha256:" in line for line in lines))
        locked = {
            line.split("==", 1)[0]: line.rsplit("--hash=sha256:", 1)[1]
            for line in lines
        }
        manifest = json.loads(flow.WHEELHOUSE_MANIFEST.read_text())
        declared = {
            item["name"]: item["sha256"]
            for item in manifest["locks"]["runtime"]["files"]
        }
        self.assertEqual(locked, declared)
        self.assertEqual(
            manifest["locks"]["runtime"]["sha256"],
            flow.sha256(flow.RUNTIME_REQUIREMENTS),
        )

    def test_runtime_requirements_install_is_bound_and_records_ase(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            prefix = run_root / "install/candidate-venv/lib/python3.12/site-packages"
            ase_path = prefix / "ase/__init__.py"
            python = run_root / "install/candidate-venv/bin/python"
            wheelhouse = run_root / "wheelhouse"
            with (
                mock.patch.object(flow, "run") as run,
                mock.patch.object(
                    flow,
                    "inspect_python_packages",
                    return_value={"ase": {"version": "3.29.0", "path": str(ase_path)}},
                ),
            ):
                identity = flow.install_runtime_requirements(
                    python, run_root, prefix, wheelhouse
                )
        command = run.call_args.args[0]
        self.assertIn("--no-index", command)
        self.assertEqual(command[command.index("--find-links") + 1], str(wheelhouse))
        self.assertIn("--no-deps", command)
        self.assertIn("--require-hashes", command)
        self.assertIn("--ignore-installed", command)
        self.assertEqual(command[-2:], ["-r", str(flow.RUNTIME_REQUIREMENTS)])
        self.assertEqual(identity["path"], "contest/config/runtime-requirements.txt")
        self.assertEqual(identity["sha256"], flow.sha256(flow.RUNTIME_REQUIREMENTS))
        self.assertEqual(identity["install_command"], command)
        self.assertEqual(identity["packages"]["ase"]["version"], "3.29.0")
        self.assertEqual(identity["packages"]["ase"]["path"], str(ase_path))

    def test_build_installs_runtime_requirements_before_identity(self):
        source = inspect.getsource(flow.build)
        self.assertLess(
            source.index("install_runtime_requirements"),
            source.index("05-candidate-identity.log"),
        )

    def test_both_requirement_installs_are_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "venv/bin/python"
            wheelhouse = root / "wheelhouse"
            with mock.patch.object(flow, "run") as run:
                commands = [
                    flow.offline_pip_install(
                        python, root, requirements, wheelhouse, root / f"{index}.log"
                    )
                    for index, requirements in enumerate(
                        (flow.BUILD_REQUIREMENTS, flow.RUNTIME_REQUIREMENTS)
                    )
                ]
        self.assertEqual(run.call_count, 2)
        for command in commands:
            self.assertIn("--no-index", command)
            self.assertEqual(command[command.index("--find-links") + 1], str(wheelhouse))
            self.assertIn("--no-deps", command)
            self.assertIn("--require-hashes", command)
            self.assertIn("--ignore-installed", command)

    def test_missing_wheelhouse_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(RuntimeError, "wheelhouse is missing"):
                flow.validate_wheelhouse(missing)
        source = inspect.getsource(flow.build)
        self.assertLess(source.index("resolve_wheelhouse"), source.index("python.is_file"))

    def test_wheelhouse_manifest_binds_lock_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            build_lock = root / "build.txt"
            runtime_lock = root / "runtime.txt"
            build_lock.write_text("build\n")
            runtime_lock.write_text("runtime\n")
            build_wheel = wheelhouse / "build.whl"
            runtime_wheel = wheelhouse / "runtime.whl"
            build_wheel.write_bytes(b"build-wheel")
            runtime_wheel.write_bytes(b"runtime-wheel")
            build_wheel_sha = flow.sha256(build_wheel)
            runtime_wheel_sha = flow.sha256(runtime_wheel)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "locks": {
                    "build": {
                        "path": "build.txt", "sha256": flow.sha256(build_lock),
                        "files": [{"name": "build", "version": "1", "filename": "build.whl", "sha256": build_wheel_sha}],
                    },
                    "runtime": {
                        "path": "runtime.txt", "sha256": flow.sha256(runtime_lock),
                        "files": [{"name": "runtime", "version": "1", "filename": "runtime.whl", "sha256": runtime_wheel_sha}],
                    },
                }
            }))
            with (
                mock.patch.object(flow, "ROOT", root),
                mock.patch.object(flow, "BUILD_REQUIREMENTS", build_lock),
                mock.patch.object(flow, "RUNTIME_REQUIREMENTS", runtime_lock),
                mock.patch.object(flow, "WHEELHOUSE_MANIFEST", manifest_path),
            ):
                identity = flow.validate_wheelhouse(wheelhouse)
        self.assertEqual(identity["path"], str(wheelhouse.resolve()))
        self.assertEqual(identity["locks"]["build"]["files"][0]["actual_sha256"], build_wheel_sha)
        self.assertEqual(identity["locks"]["runtime"]["files"][0]["actual_sha256"], runtime_wheel_sha)

    def test_contest_git_has_no_binary_wheels(self):
        tracked = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parents[2]), "ls-files", "contest"],
            text=True,
        ).splitlines()
        self.assertFalse([name for name in tracked if Path(name).suffix == ".whl"])

    def test_two_disjoint_public_pairs_and_order(self):
        with mock.patch.object(benchmark, "load_atoms", return_value=FakeAtoms()):
            pairs = benchmark.generate_sequences(Path("unused"), warmup=2, measured=3)
        self.assertEqual(["".join("A" if r == "baseline" else "B" for r in order)
                          for order in benchmark.PAIR_ORDERS], ["AB", "BA"])
        self.assertEqual(len(pairs), 2)
        hashes = [digest for pair in pairs for digest in pair["frame_hashes"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertTrue(all(not set(pair["frame_hashes"][:2]) &
                            set(pair["frame_hashes"][2:]) for pair in pairs))
        self.assertEqual(benchmark.PUBLIC_SEEDS, (510421, 620531, 730637))
        with self.assertRaisesRegex(RuntimeError, "legacy 3-pair"):
            benchmark.generate_sequences(Path("unused"), warmup=2, measured=3, pair_count=3)

    def test_quick_sequence_is_one_pair_with_short_protocol(self):
        with mock.patch.object(benchmark, "load_atoms", return_value=FakeAtoms()):
            pairs = benchmark.generate_sequences(
                Path("unused"), warmup=2, measured=10, pair_count=1
            )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(pairs[0]["frames"]), 12)
        self.assertEqual(pairs[0]["seed"], 510421)

    def test_correctness_status_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            actual = root / "actual.npz"
            fields = {
                "measured_energy_eV": np.zeros(2, dtype=np.float64),
                "measured_forces": np.zeros((2, 1024, 3), dtype=np.float64),
                "measured_virial": np.zeros((2, 3, 3), dtype=np.float64),
                "measured_stress": np.zeros((2, 6), dtype=np.float64),
                "warmup_latencies_s": np.ones(2, dtype=np.float64),
                "measured_latencies_s": np.ones(2, dtype=np.float64),
            }
            np.savez(reference, **fields)
            broken = dict(fields)
            broken["measured_energy_eV"] = np.ones(2, dtype=np.float64)
            np.savez(actual, **broken)
            tolerance_path = Path(__file__).parents[1] / "config/benchmark-tolerance.json"
            tolerance = benchmark.load_tolerance(tolerance_path)
            baseline = benchmark.validate_output(actual, reference, root / "b.json",
                                                  "baseline", "Pair1", tolerance,
                                                  tolerance_path)
            candidate = benchmark.validate_output(actual, reference, root / "c.json",
                                                   "candidate", "Pair1", tolerance,
                                                   tolerance_path)
            self.assertEqual(baseline["status"], "BENCHMARK_INVALID")
            self.assertEqual(candidate["status"], "CANDIDATE_INVALID")

    def test_output_contract_matches_historical_nano_evidence(self):
        contract = benchmark.load_output_contract()
        self.assertEqual(contract["measured_dimension"], "N")
        self.assertEqual(contract["atom_count"], 1024)
        self.assertEqual(set(contract["fields"]), benchmark.ARCHIVE_FIELDS)
        self.assertEqual(contract["fields"]["warmup_latencies_s"]["shape"], ["W"])
        self.assertEqual(contract["fields"]["measured_latencies_s"]["shape"], ["N"])
        for spec in contract["fields"].values():
            self.assertEqual(spec["allowed_dtypes"], ["float64"])
        historical = json.loads((
            Path(__file__).parents[2].parent
            / "dpa4c-nano-rc4-final-regression-20260915-evidence"
            / "results/public-benchmark/Pair1/baseline.correctness.json"
        ).read_text())
        self.assertEqual(historical["checks"]["measured_energy_eV"]["shape"], [500])
        self.assertEqual(historical["checks"]["measured_forces"]["shape"], [500, 1024, 3])
        self.assertEqual(historical["checks"]["measured_virial"]["shape"], [500, 3, 3])
        self.assertEqual(historical["checks"]["measured_stress"]["shape"], [500, 6])

    def test_output_contract_rejects_wrong_shape_dtype_fields_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tolerance_path = Path(__file__).parents[1] / "config/benchmark-tolerance.json"
            tolerance = benchmark.load_tolerance(tolerance_path)
            reference_fields = valid_output_fields()
            np.savez(root / "reference.npz", **reference_fields)

            wrong_shape = valid_output_fields()
            wrong_shape["measured_forces"] = np.zeros((2, 1), dtype=np.float64)
            np.savez(root / "wrong-shape.npz", **wrong_shape)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "wrong-shape.npz", root / "reference.npz",
                                          root / "wrong-shape.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            wrong_dtype = valid_output_fields(dtype=np.float32)
            np.savez(root / "wrong-dtype.npz", **wrong_dtype)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "wrong-dtype.npz", root / "reference.npz",
                                          root / "wrong-dtype.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            missing = valid_output_fields()
            missing.pop("measured_stress")
            np.savez(root / "missing.npz", **missing)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "missing.npz", root / "reference.npz",
                                          root / "missing.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            extra = valid_output_fields()
            extra["extra"] = np.zeros(2, dtype=np.float64)
            np.savez(root / "extra.npz", **extra)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "extra.npz", root / "reference.npz",
                                          root / "extra.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            nan = valid_output_fields()
            nan["measured_energy_eV"][0] = np.nan
            np.savez(root / "nan.npz", **nan)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "nan.npz", root / "reference.npz",
                                          root / "nan.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            bad_reference = valid_output_fields()
            bad_reference["measured_forces"] = np.zeros((2, 1), dtype=np.float64)
            np.savez(root / "bad-reference.npz", **bad_reference)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "reference.npz", root / "bad-reference.npz",
                                          root / "bad-reference.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "BENCHMARK_INVALID")

            timing_shape = valid_output_fields(warmup=3)
            np.savez(root / "timing-shape.npz", **timing_shape)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "timing-shape.npz", root / "reference.npz",
                                          root / "timing-shape.json", "candidate", "Pair1",
                                          tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            timing_dtype = valid_output_fields(dtype=np.float64)
            timing_dtype["measured_latencies_s"] = np.ones(2, dtype=np.float32)
            np.savez(root / "timing-dtype.npz", **timing_dtype)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "timing-dtype.npz", root / "reference.npz",
                                          root / "timing-dtype.json", "candidate", "Pair1",
                                          tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            timing_missing = valid_output_fields()
            timing_missing.pop("warmup_latencies_s")
            np.savez(root / "timing-missing.npz", **timing_missing)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "timing-missing.npz", root / "reference.npz",
                                          root / "timing-missing.json", "candidate", "Pair1",
                                          tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            timing_nan = valid_output_fields()
            timing_nan["measured_latencies_s"][0] = np.nan
            np.savez(root / "timing-nan.npz", **timing_nan)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "timing-nan.npz", root / "reference.npz",
                                          root / "timing-nan.json", "candidate", "Pair1",
                                          tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            timing_inf = valid_output_fields()
            timing_inf["measured_latencies_s"][0] = np.inf
            np.savez(root / "timing-inf.npz", **timing_inf)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "timing-inf.npz", root / "reference.npz",
                                          root / "timing-inf.json", "candidate", "Pair1",
                                          tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            timing_nonpositive = valid_output_fields()
            timing_nonpositive["warmup_latencies_s"][0] = 0.0
            np.savez(root / "timing-nonpositive.npz", **timing_nonpositive)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "timing-nonpositive.npz", root / "reference.npz",
                                          root / "timing-nonpositive.json", "candidate", "Pair1",
                                          tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(raised.exception.status, "CANDIDATE_INVALID")

            legal = valid_output_fields()
            np.savez(root / "legal.npz", **legal)
            legal_result = benchmark.validate_output(root / "legal.npz", root / "reference.npz",
                                                     root / "legal.json", "candidate", "Pair1",
                                                     tolerance, tolerance_path, warmup=2, measured=2)
            self.assertEqual(legal_result["status"], "PASS")

            both_bad = valid_output_fields()
            both_bad["measured_forces"] = np.zeros((2, 1), dtype=np.float64)
            np.savez(root / "both-bad.npz", **both_bad)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "both-bad.npz", root / "both-bad.npz",
                                          root / "both-bad.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "BENCHMARK_INVALID")

            bad_reference_nan = valid_output_fields()
            bad_reference_nan["measured_energy_eV"][0] = np.nan
            np.savez(root / "bad-reference-nan.npz", **bad_reference_nan)
            with self.assertRaises(benchmark.RouteFailure) as raised:
                benchmark.validate_output(root / "reference.npz", root / "bad-reference-nan.npz",
                                          root / "bad-reference-nan.json", "candidate", "Pair1",
                                          tolerance, tolerance_path)
            self.assertEqual(raised.exception.status, "BENCHMARK_INVALID")

    def test_smoke_and_benchmark_tolerances_are_separate(self):
        runtime = json.loads((Path(__file__).parents[1] / "config/runtime.json").read_text())
        dynamic = benchmark.load_tolerance(
            Path(__file__).parents[1] / "config/benchmark-tolerance.json"
        )
        self.assertEqual(runtime["tolerances"], {
            "rtol": 0.0, "energy_eV": 0.00034, "forces_eV_per_A": 0.000055,
            "virial_eV": 0.0005, "stress_eV_per_A3": 1e-7,
        })
        self.assertEqual(dynamic["fields"]["energy_eV"]["atol"],
                         0.00039577481061314757)
        self.assertNotEqual(runtime["tolerances"]["energy_eV"],
                            dynamic["fields"]["energy_eV"]["atol"])

    def test_rtol_formula_uses_reference_magnitude(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tolerance_path = Path(__file__).parents[1] / "config/benchmark-tolerance.json"
            tolerance = benchmark.load_tolerance(tolerance_path)
            atol = tolerance["fields"]["energy_eV"]["atol"]
            rtol = tolerance["fields"]["energy_eV"]["rtol"]
            reference_fields = {
                "measured_energy_eV": np.asarray([1.0e10], dtype=np.float64),
                "measured_forces": np.zeros((1, 1024, 3), dtype=np.float64),
                "measured_virial": np.zeros((1, 3, 3), dtype=np.float64),
                "measured_stress": np.zeros((1, 6), dtype=np.float64),
                "warmup_latencies_s": np.ones(1, dtype=np.float64),
                "measured_latencies_s": np.ones(1, dtype=np.float64),
            }
            np.savez(root / "reference.npz", **reference_fields)
            inside = dict(reference_fields)
            inside["measured_energy_eV"] = np.asarray(
                [1.0e10 + 0.99 * (atol + rtol * 1.0e10)], dtype=np.float64
            )
            np.savez(root / "inside.npz", **inside)
            result = benchmark.validate_output(root / "inside.npz", root / "reference.npz",
                                               root / "inside.json", "candidate", "Pair1",
                                               tolerance, tolerance_path)
            self.assertEqual(result["status"], "PASS")
            outside = dict(reference_fields)
            outside["measured_energy_eV"] = np.asarray(
                [1.0e10 + 1.01 * (atol + rtol * 1.0e10)], dtype=np.float64
            )
            np.savez(root / "outside.npz", **outside)
            result = benchmark.validate_output(root / "outside.npz", root / "reference.npz",
                                               root / "outside.json", "candidate", "Pair1",
                                               tolerance, tolerance_path)
            self.assertEqual(result["status"], "CANDIDATE_INVALID")

    def test_binding_tolerance_identity(self):
        identity = flow.benchmark_tolerance_identity()
        self.assertEqual(identity["tolerance_id"],
                         "nano-1024-fp32-public-dynamic-baseline-v1")
        self.assertEqual(identity["sha256"], flow.sha256(flow.BENCHMARK_TOLERANCE))
        self.assertEqual(identity["fields"]["virial_eV"]["atol"],
                         0.0006802242146053405)
        self.assertEqual(identity["output_contract"]["atom_count"], 1024)
        self.assertEqual(identity["output_contract"]["warmup_dimension"], "W")
        self.assertEqual(identity["output_contract"]["sha256"], flow.sha256(flow.OUTPUT_CONTRACT))

    def _package_fixture(self, root: Path):
        run_root = root / "run-root"
        benchmark_root = run_root / "results/public-benchmark"
        benchmark_root.mkdir(parents=True)
        result = benchmark_root / "result.json"
        repeats = benchmark_root / "repeats.json"
        result.write_text('{"status":"PASS"}\n')
        repeats.write_text('{"status":"PASS"}\n')
        binding = {
            "status": "PASS", "profile": "full", "verified": False,
            "source": {"commit": "candidate", "tree": "tree"},
            "starter": {"ref": "base", "resolved_commit": "starter"},
            "protocol": {
                "version": "dpa4c-ppu-contest.public-benchmark.v2",
                "warmup": 20, "measured": 100, "pairs": 2,
                "order": ["AB", "BA"],
                "aggregation_method": "paired_geometric_mean_speedup",
            },
            "result": {"path": str(result), "sha256": flow.sha256(result)},
            "output_artifacts": {
                "result.json": flow.sha256(result),
                "repeats.json": flow.sha256(repeats),
            },
        }
        (benchmark_root / "measurement-binding.json").write_text(json.dumps(binding))
        (benchmark_root / "BENCHMARK_STATUS.json").write_text('{"status":"PASS"}\n')
        return run_root, benchmark_root, binding

    def test_package_artifact_binding_rejects_drift_before_submission_creation(self):
        variants = ("result", "repeats", "missing", "binding", "escape")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_root, benchmark_root, binding = self._package_fixture(root)
                if variant == "result":
                    (benchmark_root / "result.json").write_text('{"status":"DRIFT"}\n')
                    message = "result.json SHA mismatch"
                elif variant == "repeats":
                    (benchmark_root / "repeats.json").write_text('{"status":"DRIFT"}\n')
                    message = "artifact SHA mismatch: repeats.json"
                elif variant == "missing":
                    (benchmark_root / "repeats.json").unlink()
                    message = "artifact is missing: repeats.json"
                elif variant == "binding":
                    (benchmark_root / "measurement-binding.json").unlink()
                    message = "No such file or directory"
                else:
                    binding["output_artifacts"]["../outside"] = "0" * 64
                    (benchmark_root / "measurement-binding.json").write_text(json.dumps(binding))
                    message = "escapes benchmark root"
                args = Namespace(profile="full", run_root=run_root,
                                 assets_root=root / "assets", starter_ref="base")
                with (
                    mock.patch.object(flow, "resolve_inputs", return_value=({}, run_root, root / "model", root / "structure")),
                    mock.patch.object(flow, "source_identity", return_value={"commit": "candidate", "tree": "tree"}),
                    mock.patch.object(flow, "git", side_effect=lambda command: "starter" if command[:2] == ["rev-parse", "base^{commit}"] else ""),
                    mock.patch.object(flow, "enforce_candidate_scope", return_value=[]),
                ):
                    if variant == "binding":
                        with self.assertRaises(FileNotFoundError):
                            flow.package(args)
                    else:
                        with self.assertRaisesRegex(RuntimeError, message):
                            flow.package(args)
                self.assertFalse((run_root / "submission").exists())

    def test_measured_tree_drift_is_rejected(self):
        status = {"status": "PASS"}
        binding = {"status": "PASS", "source": {"commit": "c", "tree": "old"},
                   "starter": {"ref": "base", "resolved_commit": "b"}}
        with self.assertRaisesRegex(RuntimeError, "measured commit/tree"):
            flow.validate_package_gate(status, binding, {"commit": "c", "tree": "new"},
                                       "base", "b")

    def test_package_rejects_legacy_three_pair_binding(self):
        binding = {
            "status": "PASS",
            "source": {"commit": "c", "tree": "t"},
            "starter": {"ref": "base", "resolved_commit": "b"},
            "profile": "full",
            "verified": False,
            "protocol": {
                "version": "dpa4c-ppu-contest.public-benchmark.v1",
                "warmup": 20, "measured": 500, "pairs": 3,
                "order": ["AB", "BA", "AB"],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "2-pair"):
            flow.validate_package_gate(
                {"status": "PASS"}, binding, {"commit": "c", "tree": "t"}, "base", "b"
            )

    def test_dirty_tree_is_rejected(self):
        with mock.patch.object(flow, "git", return_value=" M contest/file"):
            with self.assertRaisesRegex(RuntimeError, "not clean"):
                flow.require_clean_committed()

    def test_binary_artifacts_are_rejected(self):
        for path in ("deepmd/lib/x.so", "build/x.o", "lib/x.a", "dist/x.whl"):
            with self.assertRaisesRegex(RuntimeError, "binary implementation"):
                flow.reject_binary_implementation_files([path])

    def test_binary_patch_clean_apply_reconstructs_same_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
            source = repo / "source.py"
            source.write_text("value = 1\n")
            subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            source.write_text("value = 2\n")
            subprocess.run(["git", "-C", str(repo), "commit", "-qam", "candidate"], check=True)
            candidate = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            patch = Path(directory) / "candidate.patch"
            result = flow.make_patch(repo, base, candidate, patch)
            expected_tree = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True
            ).strip()
            self.assertEqual(result["reconstructed_tree"], expected_tree)
            self.assertEqual(result["changed_files"], ["source.py"])

    def test_candidate_scope_rejects_added_deleted_renamed_and_mode_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
            (repo / "source.py").write_text("value = 1\n")
            (repo / "contest/candidate").mkdir(parents=True)
            (repo / "contest/scripts").mkdir(parents=True)
            (repo / "contest/config").mkdir(parents=True)
            (repo / "contest/candidate/old.py").write_text("candidate\n")
            (repo / "contest/scripts/old.py").write_text("script\n")
            (repo / "contest/config/old.json").write_text("{}\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            (repo / "source.py").chmod(0o755)
            (repo / "contest/scripts/new.py").write_text("new\n")
            (repo / "contest/config/old.json").unlink()
            subprocess.run(
                ["git", "-C", str(repo), "mv", "contest/candidate/old.py", "contest/scripts/renamed.py"],
                check=True,
            )
            subprocess.run(["git", "-C", str(repo), "add", "-u", "source.py", "contest/config/old.json"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "contest/scripts/new.py"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "scope"], check=True)
            candidate = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            files = flow.changed_files(repo, base, candidate)
            self.assertIn("source.py", files)
            self.assertIn("contest/scripts/new.py", files)
            self.assertIn("contest/config/old.json", files)
            self.assertIn("contest/candidate/old.py", files)
            self.assertIn("contest/scripts/renamed.py", files)
            with self.assertRaisesRegex(RuntimeError, "contest/scripts/new.py"):
                flow.enforce_candidate_scope(base, candidate, repo)

    def test_candidate_scope_allows_project_and_candidate_changes(self):
        self.assertEqual(
            flow.candidate_scope_violations(
                ["source/module.cpp", "CMakeLists.txt", "contest/candidate/session.py"]
            ),
            [],
        )

    def test_candidate_scope_rejects_all_protected_contest_paths(self):
        protected = [
            "README.md",
            "contest/scripts/new.py", "contest/config/new.json", "contest/contest.sh",
            "contest/tests/new_test.py", "contest/image/Dockerfile", "contest/README.md",
        ]
        violations = flow.candidate_scope_violations(protected)
        self.assertEqual(violations, sorted(protected))

    def test_all_defaults_are_unique_and_stage_commands_require_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(flow, "DEFAULT_RUNS_ROOT", root / "runs"),
                mock.patch.dict(os.environ, {"DPA4C_OWNER": "owner/test"}, clear=False),
            ):
                first = flow.new_run_root("quick")
                second = flow.new_run_root("quick")
            self.assertNotEqual(first, second)
            self.assertIn("quick", first.name)
            self.assertEqual(first.parent.name, "owner-test")
            self.assertTrue(first.is_dir() and second.is_dir())
        with self.assertRaisesRegex(RuntimeError, "requires explicit --run-root"):
            flow.execute_command(Namespace(command="build", profile="quick"))
        with self.assertRaisesRegex(RuntimeError, "requires explicit --assets-root"):
            flow.execute_command(Namespace(command="build", profile="quick", run_root=Path("/run")))

    def test_all_explicit_paths_take_priority_over_auto_run_root(self):
        args = Namespace(command="all", profile="quick", starter_ref="base",
                          run_root=Path("/explicit/run"), assets_root=Path("/explicit/assets"))
        with (
            mock.patch.object(flow, "new_run_root", side_effect=AssertionError("auto root used")),
            mock.patch.object(flow, "build"),
            mock.patch.object(flow, "test"),
            mock.patch.object(flow, "benchmark"),
            mock.patch.object(flow, "candidate_change_status", return_value={"candidate_change": "NO_CANDIDATE_CHANGE"}),
            mock.patch.object(flow, "write_flow_status"),
        ):
            flow.execute_command(args)
        self.assertEqual(args.run_root, Path("/explicit/run"))
        self.assertEqual(args.assets_root, Path("/explicit/assets"))

    def test_all_order(self):
        calls = []
        args = Namespace(command="all", profile="full", starter_ref="base",
                          run_root=Path("/tmp/flow-test"))
        with (mock.patch.object(flow, "build", side_effect=lambda _: calls.append("build")),
              mock.patch.object(flow, "test", side_effect=lambda _: calls.append("test")),
              mock.patch.object(flow, "benchmark", side_effect=lambda _: calls.append("benchmark")),
              mock.patch.object(flow, "package", side_effect=lambda _: calls.append("package")),
              mock.patch.object(flow, "candidate_change_status",
                                return_value={"candidate_change": "CANDIDATE_CHANGE_PRESENT"}),
              mock.patch.object(flow, "write_flow_status")):
            flow.execute_command(args)
        self.assertEqual(calls, ["build", "test", "benchmark", "package"])

    def test_profiles_are_fixed_and_full_uses_two_pair_protocol(self):
        self.assertEqual(flow.profile_config("quick"), {
            "pair_count": 1, "pair_order": ["AB"], "warmup": 2, "measured": 10,
            "score_type": "development_quick", "verified": False,
            "package": False, "aggregation_method": "paired_speedup",
        })
        self.assertEqual(flow.profile_config("full"), {
            "pair_count": 2, "pair_order": ["AB", "BA"], "warmup": 20, "measured": 100,
            "score_type": "public_self_test", "verified": False,
            "package": True, "aggregation_method": "paired_geometric_mean_speedup",
        })

    def test_all_quick_skips_package_and_writes_status(self):
        calls = []
        args = Namespace(command="all", profile="quick", starter_ref="base",
                          run_root=Path("/tmp/flow-test"))
        with (mock.patch.object(flow, "build", side_effect=lambda _: calls.append("build")),
              mock.patch.object(flow, "test", side_effect=lambda _: calls.append("test")),
              mock.patch.object(flow, "benchmark", side_effect=lambda _: calls.append("benchmark")),
              mock.patch.object(flow, "package", side_effect=lambda _: calls.append("package")),
              mock.patch.object(flow, "candidate_change_status",
                                return_value={"candidate_change": "NO_CANDIDATE_CHANGE"}),
              mock.patch.object(flow, "write_flow_status") as write_status):
            flow.execute_command(args)
        self.assertEqual(calls, ["build", "test", "benchmark"])
        payload = write_status.call_args.args[1]
        self.assertEqual(payload["package"], "PACKAGE_SKIPPED")

    def test_package_rejects_quick_before_reading_run_root(self):
        args = Namespace(profile="quick")
        with self.assertRaisesRegex(RuntimeError, "PACKAGE_SKIPPED"):
            flow.package(args)

    def test_build_reuse_requires_exact_identity_and_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_python = root / "install/candidate-venv/bin/python"
            wheel = root / "build/wheels/deepmd_kit.whl"
            entry = root / "install/site-packages/dpa4c_candidate"
            candidate_python.parent.mkdir(parents=True)
            wheel.parent.mkdir(parents=True)
            entry.mkdir(parents=True)
            candidate_python.write_text("python\n")
            wheel.write_text("wheel\n")
            expected = {"source": {"commit": "c", "tree": "t"}, "deps": "d"}
            status = {
                "status": "PASS", "build_inputs": expected,
                "candidate_python": str(candidate_python),
                "wheel": {"path": str(wheel)},
                "candidate_entry": {"path": str(entry)},
            }
            results = root / "results"
            results.mkdir()
            (results / "BUILD_STATUS.json").write_text(json.dumps(status))
            reused = flow.reusable_build_status(root, expected)
            self.assertEqual(reused["build_inputs"], expected)
            self.assertTrue((results / "BUILD_REUSE.json").is_file())
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                flow.reusable_build_status(root, {"source": {"commit": "new", "tree": "t"}, "deps": "d"})

    def test_contest_entrypoint_resolves_repository_symlink(self):
        repo = Path(__file__).parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "dpa4c-contestant-flow"
            link.symlink_to(repo / "contest/contest.sh")
            result = subprocess.run(
                [str(link), "image", "--run-root", str(root / "run"),
                 "--assets-root", str(root / "assets")],
                text=True, capture_output=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing model:", result.stderr)
        self.assertNotIn("can't open file", result.stderr)

    def test_rc6_is_the_only_default_starter(self):
        self.assertEqual(flow.DEFAULT_STARTER_REF, "dpa4c-ppu-nano-starter-v1.0.0-rc6")
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        self.assertIn("dpa4c-ppu-nano-starter-v1.0.0-rc6", readme)

    def test_contestant_onboarding_docs_bind_image_entrypoint_and_protocols(self):
        contest_readme = (Path(__file__).parents[1] / "README.md").read_text()
        root_readme = (Path(__file__).parents[2] / "README.md").read_text()
        self.assertIn("/opt/dpa4c-contestant-kit", contest_readme)
        self.assertIn("candidate/my-model", contest_readme)
        self.assertIn("all --profile quick", contest_readme)
        self.assertIn("all --profile full", contest_readme)
        self.assertIn("1 pair", contest_readme)
        self.assertIn("2 fresh pairs", contest_readme)
        self.assertIn("20 warmup + 500 measured", contest_readme)
        self.assertIn("--run-root", contest_readme)
        self.assertIn("contest/README.md", root_readme)

    def test_runtime_image_binds_rc6_tag_and_external_provenance(self):
        dockerfile = (Path(__file__).parents[1] / "image/Dockerfile").read_text()
        runtime = json.loads((Path(__file__).parents[1] / "config/runtime.json").read_text())
        self.assertIn("--branch dpa4c-ppu-nano-starter-v1.0.0-rc6", dockerfile)
        self.assertEqual(runtime["baseline_python"], "/opt/dpa4c-baseline-venv/bin/python")
        self.assertIn("DPA4C_BASELINE_VENV=/opt/dpa4c-baseline-venv", dockerfile)
        self.assertIn("DPA4C_BASELINE_BUILD=/opt/dpa4c-baseline-build", dockerfile)
        self.assertIn("python -m venv --system-site-packages", dockerfile)
        self.assertIn('baseline_python="$DPA4C_BASELINE_VENV/bin/python"', dockerfile)
        self.assertIn('pip wheel "$DPA4C_CONTESTANT_ROOT"', dockerfile)
        self.assertIn('"baseline_wheel"', dockerfile)
        self.assertIn("describe --exact-match --tags", dockerfile)
        self.assertIn("symbolic-ref -q HEAD", dockerfile)
        self.assertIn("DPA4C_RESOLVED_HEAD", dockerfile)
        self.assertIn("DPA4C_RESOLVED_TREE", dockerfile)
        self.assertIn("/opt/dpa4c-baseline-identity.json", dockerfile)
        self.assertIn("import deepmd.lib", dockerfile)
        self.assertIn("from deepmd.calculator import DP", dockerfile)
        self.assertIn("env -u PYTHONPATH -u PYTHONHOME", dockerfile)
        self.assertIn("baseline deepmd.lib contains no shared objects", dockerfile)
        self.assertNotIn("com.dptech.dpa4c.source-commit", dockerfile)
        self.assertNotIn("com.dptech.dpa4c.source-tree", dockerfile)

    def test_legacy_entry_is_thin_forwarder(self):
        legacy = Path(__file__).parents[2] / "examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh"
        text = legacy.read_text()
        self.assertIn('exec "$REPO_ROOT/contest/contest.sh" "$@"', text)
        self.assertNotIn("pip ", text)

if __name__ == "__main__":
    unittest.main()
