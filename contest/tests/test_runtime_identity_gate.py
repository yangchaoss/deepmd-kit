from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


FLOW_PATH = Path(__file__).parents[1] / "scripts" / "flow.py"
SPEC = importlib.util.spec_from_file_location("contest_flow", FLOW_PATH)
flow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(flow)


class RuntimeIdentityGateTest(unittest.TestCase):
    def fixture(self, root: Path):
        prefix = root / "candidate-prefix"
        entry = prefix / "dpa4c_candidate"
        entry.mkdir(parents=True)
        init = entry / "__init__.py"
        session = entry / "session.py"
        init.write_text("from .session import create_session\n")
        session.write_text("def create_session(): pass\n")
        candidate = {
            "status": "PASS",
            "deepmd": str(prefix / "deepmd/__init__.py"),
            "deepmd_lib": str(prefix / "deepmd/lib/__init__.py"),
            "deepmd_elfs": [
                {"path": str(prefix / "deepmd/lib/libdeepmd.so"), "sha256": "a", "size_bytes": 1}
            ],
            "torch": {"version": flow.FROZEN_TORCH_VERSION, "path": "/opt/ac2/torch/__init__.py"},
            "candidate_entry": str(init),
        }
        baseline = {
            "status": "PASS",
            "deepmd": "/opt/ac2/deepmd/__init__.py",
            "deepmd_lib": "/opt/ac2/deepmd/lib/__init__.py",
            "deepmd_elfs": [
                {"path": "/opt/ac2/deepmd/lib/libdeepmd.so", "sha256": "b", "size_bytes": 2}
            ],
            "torch": {"version": flow.FROZEN_TORCH_VERSION, "path": "/opt/ac2/torch/__init__.py"},
        }
        status = {
            "status": "PASS",
            "source": {"commit": "c", "tree": "t"},
            "candidate_python": str(root / "candidate-python"),
            "candidate_prefix": str(prefix),
            "baseline_prefix": "/opt/ac2",
            "candidate_entry": {
                "path": str(entry),
                "files": {"__init__.py": flow.sha256(init), "session.py": flow.sha256(session)},
            },
            "runtime_identity": {"candidate": candidate, "baseline": baseline},
        }
        return status, candidate, baseline

    def test_correct_identity_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            status, candidate, baseline = self.fixture(Path(directory))
            result = flow.validate_runtime_identities(
                status, candidate, baseline, candidate_returncode=0, baseline_returncode=0
            )
            self.assertEqual(result["status"], "PASS")

    def test_wrong_candidate_is_rejected_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            (run_root / "results").mkdir(parents=True)
            (run_root / "logs").mkdir()
            status, candidate, baseline = self.fixture(root)
            (run_root / "results" / "BUILD_STATUS.json").write_text(json.dumps(status))
            (run_root / "results" / "SMOKE_STATUS.json").write_text('{"status":"PASS"}')
            wrong = dict(candidate)
            wrong.update({"status": "FAIL", "deepmd": "/opt/ac2/deepmd/__init__.py"})
            calls = []

            def fake_identity(*args, output, **kwargs):
                return (2, wrong) if "candidate" in output.name else (0, baseline)

            args = Namespace(run_root=run_root, assets_root=root, baseline_python=None)
            with (
                mock.patch.object(flow, "resolve_inputs", return_value=({"baseline_python": "/opt/ac2/bin/python"}, run_root, root / "model", root / "structure")),
                mock.patch.object(flow, "require_clean_committed", return_value=status["source"]),
                mock.patch.object(flow, "run_identity", side_effect=fake_identity),
                mock.patch.object(flow, "run", side_effect=lambda *a, **k: calls.append((a, k))),
            ):
                with self.assertRaisesRegex(RuntimeError, "CANDIDATE_INVALID"):
                    flow.test(args)
            self.assertEqual(calls, [])
            self.assertFalse((run_root / "results" / "SMOKE_STATUS.json").exists())
            gate = json.loads((run_root / "results" / "RUNTIME_IDENTITY_STATUS.json").read_text())
            self.assertEqual(gate["status"], "CANDIDATE_INVALID")
            self.assertFalse(gate["workers_started"])

    def test_wrong_baseline_is_benchmark_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            status, candidate, baseline = self.fixture(Path(directory))
            wrong = dict(baseline)
            wrong["deepmd"] = "/wrong/deepmd/__init__.py"
            result = flow.validate_runtime_identities(
                status, candidate, wrong, candidate_returncode=0, baseline_returncode=0
            )
            self.assertEqual(result["status"], "BENCHMARK_INVALID")


if __name__ == "__main__":
    unittest.main()
