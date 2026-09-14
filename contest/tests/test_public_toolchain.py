from __future__ import annotations

import importlib.util
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


class PublicBenchmarkTest(unittest.TestCase):
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
            tolerances = {"energy_eV": 3.4e-4, "forces_eV_per_A": 5.5e-5,
                          "virial_eV": 5e-4, "stress_eV_per_A3": 1e-7}
            baseline = benchmark.validate_output(actual, reference, root / "b.json",
                                                  "baseline", "Pair1", tolerances)
            candidate = benchmark.validate_output(actual, reference, root / "c.json",
                                                   "candidate", "Pair1", tolerances)
            self.assertEqual(baseline["status"], "BENCHMARK_INVALID")
            self.assertEqual(candidate["status"], "CANDIDATE_INVALID")

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

    def test_legacy_entry_is_thin_forwarder(self):
        legacy = Path(__file__).parents[2] / "examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh"
        text = legacy.read_text()
        self.assertIn('exec "$REPO_ROOT/contest/contest.sh" "$@"', text)
        self.assertNotIn("pip ", text)

if __name__ == "__main__":
    unittest.main()
