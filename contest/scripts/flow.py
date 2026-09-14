#!/usr/bin/env python3
"""Single organizer-facing build/test/package/image entrypoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTEST = ROOT / "contest"
CONFIG = CONTEST / "config" / "runtime.json"
BUILD_REQUIREMENTS = CONTEST / "config" / "build-requirements.txt"
FROZEN_TORCH_VERSION = "2.9.0+ali.10.ppu2.1.0.cu130"
FROZEN_TORCH_ROOT = Path("/opt/ac2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], *, cwd: Path, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    (log.with_suffix(log.suffix + ".returncode")).write_text(f"{proc.returncode}\n")
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log}")


def run_identity(
    command: list[str], *, cwd: Path, log: Path, env: dict[str, str], output: Path
) -> tuple[int, dict[str, object]]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    (log.with_suffix(log.suffix + ".returncode")).write_text(f"{proc.returncode}\n")
    if not output.is_file():
        return proc.returncode, {"status": "FAIL", "error": "identity output missing"}
    try:
        return proc.returncode, json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return proc.returncode or 2, {
            "status": "FAIL",
            "error": f"invalid identity output: {exc}",
        }


def clean_env(python: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
        "PYTHONUSERBASE", "PYTHONSTARTUP", "LD_PRELOAD", "LD_LIBRARY_PATH", "LIBRARY_PATH",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PATH": str(python.absolute().parent) + os.pathsep + env.get("PATH", ""),
            "PIP_NO_CACHE_DIR": "1",
            "CMAKE_BUILD_PARALLEL_LEVEL": "8",
            "PPU_SDK": "/usr/local/PPU_SDK",
            "CUDA_HOME": "/usr/local/PPU_SDK/CUDA_SDK",
        }
    )
    return env


def git(command: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *command], text=True).strip()


def require_clean_committed() -> dict[str, str]:
    status = git(["status", "--porcelain"])
    if status:
        raise RuntimeError(f"candidate checkout is not clean:\n{status}")
    return {"commit": git(["rev-parse", "HEAD"]), "tree": git(["rev-parse", "HEAD^{tree}"])}


def resolve_inputs(args) -> tuple[dict[str, object], Path, Path, Path]:
    config = json.loads(CONFIG.read_text())
    run_root = args.run_root.resolve()
    try:
        run_root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("run root must be outside the Git checkout")
    assets = args.assets_root.resolve()
    model = assets / config["assets"]["model"]["name"]
    structure = assets / config["assets"]["structure"]["name"]
    for name, path in (("model", model), ("structure", structure)):
        if not path.is_file():
            raise RuntimeError(f"missing {name}: {path}")
        expected = config["assets"][name]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA mismatch: {actual} != {expected}")
    return config, run_root, model, structure


def candidate_python(run_root: Path) -> Path:
    return run_root / "install" / "candidate-venv" / "bin" / "python"


def site_packages(python: Path, run_root: Path) -> Path:
    out = subprocess.check_output(
        [str(python), "-s", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        cwd=run_root,
        env=clean_env(python),
        text=True,
    ).strip()
    return Path(out).resolve()


def inspect_python_packages(
    python: Path, run_root: Path, packages: tuple[tuple[str, str], ...]
) -> dict[str, object]:
    package_list = repr(packages)
    code = f"""
import importlib
import importlib.metadata
import json
import pathlib
items = {{}}
for module_name, distribution in {package_list}:
    module = importlib.import_module(module_name)
    items[module_name] = {{
        'version': importlib.metadata.version(distribution),
        'path': str(pathlib.Path(module.__file__).resolve()),
    }}
print(json.dumps(items))
"""
    output = subprocess.check_output(
        [str(python), "-s", "-c", code], cwd=run_root, env=clean_env(python), text=True
    )
    return json.loads(output)


def inspect_build_environment(python: Path, run_root: Path) -> dict[str, object]:
    packages = (
        ("scikit_build_core", "scikit-build-core"),
        ("dependency_groups", "dependency-groups"),
        ("pathspec", "pathspec"),
        ("packaging", "packaging"),
        ("setuptools_scm", "setuptools-scm"),
        ("hatch_fancy_pypi_readme", "hatch-fancy-pypi-readme"),
        ("setuptools", "setuptools"),
        ("torch", "torch"),
    )
    result = inspect_python_packages(python, run_root, packages)
    torch = result["torch"]
    if torch["version"] != FROZEN_TORCH_VERSION:
        raise RuntimeError(f"unexpected build-time Torch version: {torch['version']}")
    try:
        Path(torch["path"]).resolve().relative_to(FROZEN_TORCH_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"build-time Torch escaped frozen root: {torch['path']}") from exc
    return result


def stable_identity(identity: dict[str, object], *, candidate: bool) -> dict[str, object]:
    fields = ("deepmd", "deepmd_lib", "deepmd_elfs", "torch")
    result = {field: identity.get(field) for field in fields}
    if candidate:
        result["candidate_entry"] = identity.get("candidate_entry")
    return result


def entry_binding_errors(
    manifest: dict[str, object], candidate_prefix: Path
) -> list[str]:
    expected_path = Path(manifest["path"]).resolve()
    actual_path = (candidate_prefix / "dpa4c_candidate").resolve()
    errors = []
    if actual_path != expected_path:
        errors.append(f"candidate entry path changed: {actual_path} != {expected_path}")
    expected_files = manifest["files"]
    actual_files = {path.name: sha256(path) for path in sorted(actual_path.glob("*.py"))}
    if actual_files != expected_files:
        errors.append("candidate entry file set or SHA changed")
    return errors


def validate_runtime_identities(
    build_status: dict[str, object],
    candidate_identity: dict[str, object],
    baseline_identity: dict[str, object],
    *,
    candidate_returncode: int,
    baseline_returncode: int,
) -> dict[str, object]:
    frozen = build_status["runtime_identity"]
    candidate_errors = entry_binding_errors(
        build_status["candidate_entry"], Path(build_status["candidate_prefix"])
    )
    if candidate_returncode or candidate_identity.get("status") != "PASS":
        candidate_errors.append("candidate pretest identity command failed")
    if stable_identity(candidate_identity, candidate=True) != stable_identity(
        frozen["candidate"], candidate=True
    ):
        candidate_errors.append("candidate stable runtime identity changed")
    baseline_errors = []
    if baseline_returncode or baseline_identity.get("status") != "PASS":
        baseline_errors.append("baseline pretest identity command failed")
    if stable_identity(baseline_identity, candidate=False) != stable_identity(
        frozen["baseline"], candidate=False
    ):
        baseline_errors.append("baseline stable runtime identity changed")
    if baseline_errors:
        status = "BENCHMARK_INVALID"
    elif candidate_errors:
        status = "CANDIDATE_INVALID"
    else:
        status = "PASS"
    return {
        "schema_version": "dpa4c-ppu-contest.runtime-identity.v1",
        "status": status,
        "workers_started": False,
        "candidate": {
            "returncode": candidate_returncode,
            "errors": candidate_errors,
            "stable_identity": stable_identity(candidate_identity, candidate=True),
        },
        "baseline": {
            "returncode": baseline_returncode,
            "errors": baseline_errors,
            "stable_identity": stable_identity(baseline_identity, candidate=False),
        },
    }


def build(args) -> None:
    config, run_root, model, structure = resolve_inputs(args)
    identity = require_clean_committed()
    for name in ("build", "install", "logs", "results", "submission"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    baseline_python = Path(args.baseline_python or config["baseline_python"])
    python = candidate_python(run_root)
    if not python.is_file():
        run([str(baseline_python), "-m", "venv", "--system-site-packages", str(python.parents[1])], cwd=run_root, log=run_root / "logs" / "01-venv.log", env=clean_env(baseline_python))
    setuptools_before = inspect_python_packages(
        python, run_root, (("setuptools", "setuptools"),)
    )["setuptools"]
    run(
        [str(python), "-s", "-m", "pip", "install", "--no-deps", "--require-hashes", "-r", str(BUILD_REQUIREMENTS)],
        cwd=run_root,
        log=run_root / "logs" / "02-build-requirements.log",
        env=clean_env(python),
    )
    build_environment = inspect_build_environment(python, run_root)
    dist = run_root / "build" / "wheels"
    dist.mkdir(exist_ok=True)
    if any(dist.iterdir()):
        raise RuntimeError(f"fresh build directory is not empty: {dist}")
    scikit_build = run_root / "build" / "scikit-build"
    if scikit_build.exists() and any(scikit_build.iterdir()):
        raise RuntimeError(f"fresh scikit-build directory is not empty: {scikit_build}")
    wheel_env = clean_env(python)
    wheel_env["SKBUILD_BUILD_DIR"] = str(scikit_build)
    wheel_env["DP_ENABLE_TENSORFLOW"] = "0"
    wheel_env["DP_ENABLE_PYTORCH"] = "1"
    run(
        [
            str(python),
            "-s",
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist),
        ],
        cwd=ROOT,
        log=run_root / "logs" / "02-wheel.log",
        env=wheel_env,
    )
    wheels = sorted(dist.glob("deepmd_kit-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one DeepMD wheel, found {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {"deepmd/lib/__init__.py", "deepmd/lib/libdeepmd.so"}
        if missing := sorted(required - names):
            raise RuntimeError(f"candidate wheel missing deepmd.lib: {missing}")
        record = next(name for name in names if name.endswith(".dist-info/RECORD"))
        record_names = {row[0] for row in csv.reader(archive.read(record).decode().splitlines())}
        if not required <= record_names:
            raise RuntimeError("wheel RECORD does not bind required deepmd.lib files")
    run([str(python), "-s", "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], cwd=run_root, log=run_root / "logs" / "03-install.log", env=clean_env(python))
    site = site_packages(python, run_root)
    entry = site / "dpa4c_candidate"
    entry.mkdir(exist_ok=False)
    for source in sorted((CONTEST / "candidate").glob("*.py")):
        shutil.copy2(source, entry / source.name)
    manifest = {
        "schema_version": "dpa4c-ppu-contest.build.v1",
        "status": "PASS",
        "source": identity,
        "wheel": {"path": str(wheel), "sha256": sha256(wheel), "size_bytes": wheel.stat().st_size},
        "candidate_python": str(python),
        "candidate_prefix": str(site),
        "candidate_entry": {"path": str(entry), "files": {p.name: sha256(p) for p in sorted(entry.glob("*.py"))}},
        "build_environment": build_environment,
        "setuptools_identity": {
            "before_build_requirements": setuptools_before,
            "after_build_requirements": build_environment["setuptools"],
        },
        "assets": {"model": {"path": str(model), "sha256": sha256(model)}, "structure": {"path": str(structure), "sha256": sha256(structure)}},
        "formal_performance": "NOT_RUN_BY_SCOPE",
    }
    run([str(python), "-s", str(CONTEST / "scripts" / "identity.py"), "--expected-prefix", str(site), "--candidate-entry", "--expected-torch-version", FROZEN_TORCH_VERSION, "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(run_root / "results" / "candidate-identity.json")], cwd=run_root, log=run_root / "logs" / "04-candidate-identity.log", env=clean_env(python))
    candidate_identity = json.loads(
        (run_root / "results" / "candidate-identity.json").read_text()
    )
    if candidate_identity["torch"] != build_environment["torch"]:
        raise RuntimeError("candidate runtime Torch identity differs from build-time Torch")
    baseline_prefix = site_packages(baseline_python, run_root)
    run([str(baseline_python), "-s", str(CONTEST / "scripts" / "identity.py"), "--expected-prefix", str(baseline_prefix), "--expected-torch-version", FROZEN_TORCH_VERSION, "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(run_root / "results" / "baseline-identity.json")], cwd=run_root, log=run_root / "logs" / "05-baseline-identity.log", env=clean_env(baseline_python))
    baseline_identity = json.loads(
        (run_root / "results" / "baseline-identity.json").read_text()
    )
    manifest["baseline_prefix"] = str(baseline_prefix)
    manifest["runtime_identity"] = {
        "candidate": candidate_identity,
        "baseline": baseline_identity,
    }
    (run_root / "results" / "BUILD_STATUS.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def test(args) -> None:
    config, run_root, model, structure = resolve_inputs(args)
    identity = require_clean_committed()
    status_path = run_root / "results" / "BUILD_STATUS.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "PASS" or status.get("source") != identity:
        raise RuntimeError("test source identity does not match successful build")
    python = Path(status["candidate_python"])
    baseline_python = Path(args.baseline_python or config["baseline_python"])
    smoke_status = run_root / "results" / "SMOKE_STATUS.json"
    smoke_status.unlink(missing_ok=True)
    candidate_pretest = run_root / "results" / "pretest-candidate-identity.json"
    baseline_pretest = run_root / "results" / "pretest-baseline-identity.json"
    candidate_rc, candidate_identity = run_identity(
        [str(python), "-s", str(CONTEST / "scripts" / "identity.py"), "--expected-prefix", status["candidate_prefix"], "--candidate-entry", "--expected-torch-version", FROZEN_TORCH_VERSION, "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(candidate_pretest)],
        cwd=run_root,
        log=run_root / "logs" / "06-pretest-candidate-identity.log",
        env=clean_env(python),
        output=candidate_pretest,
    )
    baseline_rc, baseline_identity = run_identity(
        [str(baseline_python), "-s", str(CONTEST / "scripts" / "identity.py"), "--expected-prefix", status["baseline_prefix"], "--expected-torch-version", FROZEN_TORCH_VERSION, "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(baseline_pretest)],
        cwd=run_root,
        log=run_root / "logs" / "07-pretest-baseline-identity.log",
        env=clean_env(baseline_python),
        output=baseline_pretest,
    )
    runtime_status = validate_runtime_identities(
        status,
        candidate_identity,
        baseline_identity,
        candidate_returncode=candidate_rc,
        baseline_returncode=baseline_rc,
    )
    runtime_status_path = run_root / "results" / "RUNTIME_IDENTITY_STATUS.json"
    runtime_status_path.write_text(json.dumps(runtime_status, indent=2) + "\n")
    if runtime_status["status"] != "PASS":
        raise RuntimeError(runtime_status["status"])
    runtime_status["workers_started"] = True
    runtime_status_path.write_text(json.dumps(runtime_status, indent=2) + "\n")
    baseline_out = run_root / "results" / "smoke-baseline"
    candidate_out = run_root / "results" / "smoke-candidate"
    run([str(baseline_python), "-s", str(CONTEST / "scripts" / "worker.py"), "--mode", "baseline", "--model", str(model), "--structure", str(structure), "--output", str(baseline_out)], cwd=run_root, log=run_root / "logs" / "08-smoke-baseline.log", env=clean_env(baseline_python))
    run([str(python), "-s", str(CONTEST / "scripts" / "worker.py"), "--mode", "candidate", "--model", str(model), "--structure", str(structure), "--output", str(candidate_out)], cwd=run_root, log=run_root / "logs" / "09-smoke-candidate.log", env=clean_env(python))
    run([str(baseline_python), "-s", str(CONTEST / "scripts" / "compare.py"), "--baseline", str(baseline_out / "efs.npz"), "--candidate", str(candidate_out / "efs.npz"), "--config", str(CONFIG), "--output", str(smoke_status)], cwd=run_root, log=run_root / "logs" / "10-compare.log", env=clean_env(baseline_python))


def unsupported(command: str) -> None:
    raise RuntimeError(f"{command} is intentionally outside A-D and is not implemented in this batch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "test", "package", "image", "all"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--baseline-python")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        elif args.command == "test":
            test(args)
        elif args.command == "all":
            build(args)
            test(args)
        else:
            unsupported(args.command)
        print(json.dumps({"status": "PASS", "command": args.command, "run_root": str(args.run_root.resolve()), "formal_performance": "NOT_RUN_BY_SCOPE"}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "command": args.command, "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
