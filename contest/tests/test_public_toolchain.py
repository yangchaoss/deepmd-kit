from __future__ import annotations

import importlib.util
import inspect
import json
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


def synthetic_aggregate() -> dict[str, object]:
    fields = {
        "measured_energy_eV": ([4], "float64", 0.001),
        "measured_forces": ([4, 1024, 3], "float64", 0.002),
        "measured_virial": ([4, 3, 3], "float64", 0.003),
        "measured_stress": ([4, 6], "float64", 0.004),
    }
    pairs = []
    for number, order in enumerate(("AB", "BA", "AB"), start=1):
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
        "schema_version": "dpa4c-ppu-contest.public-aggregate.v1",
        "status": "PASS", "score_type": "public_self_test", "verified": False,
        "paired_median_speedup": 1.02,
        "benchmark_tolerance": {"fields": {"energy_eV": {"atol": 0.01, "rtol": 0.0}}},
        "pair_speedups_candidate_over_baseline": [1.01, 1.02, 1.03], "pairs": pairs,
    }


class PublicBenchmarkTest(unittest.TestCase):
    def test_compact_summary_aggregates_records_and_keeps_repeats_separate(self):
        aggregate = synthetic_aggregate()
        compact = benchmark.compact_public_result(aggregate, warmup=2, measured=4)
        self.assertEqual(compact["status"], "PASS")
        self.assertEqual(compact["result_format"], "compact")
        self.assertEqual(compact["protocol"], {
            "pair_order": ["AB", "BA", "AB"], "pair_count": 3,
            "warmup": 2, "measured": 4,
        })
        correctness = compact["correctness_summary"]
        self.assertEqual(correctness["status"], "PASS")
        self.assertEqual(correctness["routes"]["candidate"]["measured_frames"], 4)
        self.assertEqual(correctness["routes"]["candidate"]["shape"]["forces_eV_per_A"], [4, 1024, 3])
        self.assertEqual(correctness["routes"]["candidate"]["max_abs"]["energy_eV"], 0.005)
        self.assertEqual(correctness["tolerance"]["energy_eV"], {"atol": 0.01, "rtol": 0.0})
        performance = compact["performance_summary"]
        self.assertEqual(performance["fresh_pair_count"], 3)
        self.assertEqual(performance["fresh_process_count"], 6)
        self.assertEqual(performance["candidate"]["median_evals_per_s"], 8.2)
        self.assertEqual(performance["candidate_latency_s"]["p50_median"], 1.11)
        self.assertEqual(performance["paired_median_speedup"], 1.02)
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

    def test_three_disjoint_public_pairs_and_order(self):
        with mock.patch.object(benchmark, "load_atoms", return_value=FakeAtoms()):
            pairs = benchmark.generate_sequences(Path("unused"), warmup=2, measured=3)
        self.assertEqual(["".join("A" if r == "baseline" else "B" for r in order)
                          for order in benchmark.PAIR_ORDERS], ["AB", "BA", "AB"])
        self.assertEqual(len(pairs), 3)
        hashes = [digest for pair in pairs for digest in pair["frame_hashes"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertTrue(all(not set(pair["frame_hashes"][:2]) &
                            set(pair["frame_hashes"][2:]) for pair in pairs))
        self.assertEqual(benchmark.PUBLIC_SEEDS, (510421, 620531, 730637))

    def test_correctness_status_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            actual = root / "actual.npz"
            fields = {
                "measured_energy_eV": np.zeros(2, dtype=np.float64),
                "measured_forces": np.zeros((2, 1024, 3), dtype=np.float32),
                "measured_virial": np.zeros((2, 3, 3), dtype=np.float32),
                "measured_stress": np.zeros((2, 6), dtype=np.float32),
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
                "measured_forces": np.zeros((1, 1, 3), dtype=np.float64),
                "measured_virial": np.zeros((1, 3, 3), dtype=np.float64),
                "measured_stress": np.zeros((1, 6), dtype=np.float64),
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

    def test_measured_tree_drift_is_rejected(self):
        status = {"status": "PASS"}
        binding = {"status": "PASS", "source": {"commit": "c", "tree": "old"},
                   "starter": {"ref": "base", "resolved_commit": "b"}}
        with self.assertRaisesRegex(RuntimeError, "measured commit/tree"):
            flow.validate_package_gate(status, binding, {"commit": "c", "tree": "new"},
                                       "base", "b")

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

    def test_all_order(self):
        calls = []
        args = Namespace(command="all")
        with (mock.patch.object(flow, "build", side_effect=lambda _: calls.append("build")),
              mock.patch.object(flow, "test", side_effect=lambda _: calls.append("test")),
              mock.patch.object(flow, "benchmark", side_effect=lambda _: calls.append("benchmark")),
              mock.patch.object(flow, "package", side_effect=lambda _: calls.append("package"))):
            flow.execute_command(args)
        self.assertEqual(calls, ["build", "test", "benchmark", "package"])

    def test_rc4_is_the_only_default_starter(self):
        self.assertEqual(flow.DEFAULT_STARTER_REF, "dpa4c-ppu-nano-starter-v1.0.0-rc4")
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        self.assertIn("dpa4c-ppu-nano-starter-v1.0.0-rc4", readme)

    def test_runtime_image_binds_rc4_tag_and_external_provenance(self):
        dockerfile = (Path(__file__).parents[1] / "image/Dockerfile").read_text()
        self.assertIn("--branch dpa4c-ppu-nano-starter-v1.0.0-rc4", dockerfile)
        self.assertIn("describe --exact-match --tags", dockerfile)
        self.assertIn("symbolic-ref -q HEAD", dockerfile)
        self.assertIn("DPA4C_RESOLVED_HEAD", dockerfile)
        self.assertIn("DPA4C_RESOLVED_TREE", dockerfile)
        self.assertNotIn("com.dptech.dpa4c.source-commit", dockerfile)
        self.assertNotIn("com.dptech.dpa4c.source-tree", dockerfile)

    def test_legacy_entry_is_thin_forwarder(self):
        legacy = Path(__file__).parents[2] / "examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh"
        text = legacy.read_text()
        self.assertIn('exec "$REPO_ROOT/contest/contest.sh" "$@"', text)
        self.assertNotIn("pip ", text)

if __name__ == "__main__":
    unittest.main()
