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
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTEST = ROOT / "contest"
CONFIG = CONTEST / "config" / "runtime.json"
BENCHMARK_TOLERANCE = CONTEST / "config" / "benchmark-tolerance.json"
BUILD_REQUIREMENTS = CONTEST / "config" / "build-requirements.txt"
RUNTIME_REQUIREMENTS = CONTEST / "config" / "runtime-requirements.txt"
CANDIDATE_MANIFEST = CONTEST / "config" / "submission.json"
FROZEN_TORCH_VERSION = "2.9.0+ali.10.ppu2.1.0.cu130"
FROZEN_TORCH_ROOT = Path("/opt/ac2")
DEFAULT_STARTER_REF = "dpa4c-ppu-nano-starter-v1.0.0-rc1"
BENCHMARK_WORKER = CONTEST / "scripts" / "public_route_worker.py"
BENCHMARK_RUNNER = CONTEST / "scripts" / "public_benchmark.py"
BINARY_IMPLEMENTATION_SUFFIXES = {".so", ".o", ".a", ".whl"}


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


def tracked_manifest_sha() -> str:
    payload = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-s", "-z"]
    )
    return hashlib.sha256(payload).hexdigest()


def benchmark_tolerance_identity() -> dict[str, object]:
    config = json.loads(BENCHMARK_TOLERANCE.read_text())
    return {
        "path": str(BENCHMARK_TOLERANCE.relative_to(ROOT)),
        "sha256": sha256(BENCHMARK_TOLERANCE),
        "tolerance_id": config["tolerance_id"],
        "fields": config["fields"],
        "scope": config["scope"],
        "interpretation": config["interpretation"],
        "derivation": config["derivation"],
        "source": config["source"],
    }


def source_identity() -> dict[str, str]:
    identity = require_clean_committed()
    identity.update({
        "repository": git(["remote", "get-url", "origin"]),
        "branch": git(["branch", "--show-current"]),
        "tracked_manifest_sha256": tracked_manifest_sha(),
    })
    return identity


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


def install_runtime_requirements(
    python: Path, run_root: Path, candidate_prefix: Path
) -> dict[str, object]:
    command = [
        str(python), "-s", "-m", "pip", "install", "--no-deps",
        "--require-hashes", "-r", str(RUNTIME_REQUIREMENTS),
    ]
    run(
        command,
        cwd=run_root,
        log=run_root / "logs" / "04-runtime-requirements.log",
        env=clean_env(python),
    )
    packages = inspect_python_packages(python, run_root, (("ase", "ase"),))
    ase = packages["ase"]
    if ase["version"] != "3.29.0":
        raise RuntimeError(f"unexpected ASE version: {ase['version']}")
    try:
        Path(ase["path"]).resolve().relative_to(candidate_prefix.resolve())
    except ValueError as exc:
        raise RuntimeError(f"ASE escaped candidate prefix: {ase['path']}") from exc
    return {
        "path": str(RUNTIME_REQUIREMENTS.relative_to(ROOT)),
        "sha256": sha256(RUNTIME_REQUIREMENTS),
        "install_command": command,
        "packages": packages,
    }


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
            "identity": candidate_identity,
            "stable_identity": stable_identity(candidate_identity, candidate=True),
        },
        "baseline": {
            "returncode": baseline_returncode,
            "errors": baseline_errors,
            "identity": baseline_identity,
            "stable_identity": stable_identity(baseline_identity, candidate=False),
        },
    }


def same_source(recorded: dict[str, object], current: dict[str, str]) -> bool:
    return all(recorded.get(key) == current[key] for key in ("commit", "tree"))


def runtime_identity_gate(
    args, build_status: dict[str, object], *, output_name: str, log_prefix: str
) -> dict[str, object]:
    config, run_root, _, _ = resolve_inputs(args)
    python = Path(build_status["candidate_python"])
    baseline_python = Path(args.baseline_python or config["baseline_python"])
    candidate_output = run_root / "results" / f"{log_prefix}-candidate-identity.json"
    baseline_output = run_root / "results" / f"{log_prefix}-baseline-identity.json"
    candidate_rc, candidate = run_identity(
        [str(python), "-s", str(CONTEST / "scripts" / "identity.py"),
         "--expected-prefix", build_status["candidate_prefix"], "--candidate-entry",
         "--expected-torch-version", FROZEN_TORCH_VERSION,
         "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(candidate_output)],
        cwd=run_root, log=run_root / "logs" / f"{log_prefix}-candidate-identity.log",
        env=clean_env(python), output=candidate_output,
    )
    baseline_rc, baseline = run_identity(
        [str(baseline_python), "-s", str(CONTEST / "scripts" / "identity.py"),
         "--expected-prefix", build_status["baseline_prefix"],
         "--expected-torch-version", FROZEN_TORCH_VERSION,
         "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(baseline_output)],
        cwd=run_root, log=run_root / "logs" / f"{log_prefix}-baseline-identity.log",
        env=clean_env(baseline_python), output=baseline_output,
    )
    result = validate_runtime_identities(
        build_status, candidate, baseline,
        candidate_returncode=candidate_rc, baseline_returncode=baseline_rc,
    )
    (run_root / "results" / output_name).write_text(json.dumps(result, indent=2) + "\n")
    return result


def build(args) -> None:
    config, run_root, model, structure = resolve_inputs(args)
    identity = source_identity()
    git(["ls-files", "--error-unmatch", str(CANDIDATE_MANIFEST.relative_to(ROOT))])
    git(["ls-files", "--error-unmatch", str(RUNTIME_REQUIREMENTS.relative_to(ROOT))])
    for name in ("build", "install", "logs", "results"):
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
    runtime_requirements = install_runtime_requirements(python, run_root, site)
    entry = site / "dpa4c_candidate"
    entry.mkdir(exist_ok=False)
    for source in sorted((CONTEST / "candidate").glob("*.py")):
        shutil.copy2(source, entry / source.name)
    manifest = {
        "schema_version": "dpa4c-ppu-contest.build.v1",
        "status": "PASS",
        "source": identity,
        "tracked_candidate_manifest": {
            "path": str(CANDIDATE_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE_MANIFEST),
        },
        "build_command": [str(ROOT / "contest" / "contest.sh"), "build"],
        "build_configuration": {
            "DP_ENABLE_TENSORFLOW": "0", "DP_ENABLE_PYTORCH": "1",
            "SKBUILD_BUILD_DIR": str(scikit_build),
            "requirements": str(BUILD_REQUIREMENTS),
            "requirements_sha256": sha256(BUILD_REQUIREMENTS),
        },
        "runtime_requirements": runtime_requirements,
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
    run([str(python), "-s", str(CONTEST / "scripts" / "identity.py"), "--expected-prefix", str(site), "--candidate-entry", "--expected-torch-version", FROZEN_TORCH_VERSION, "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(run_root / "results" / "candidate-identity.json")], cwd=run_root, log=run_root / "logs" / "05-candidate-identity.log", env=clean_env(python))
    candidate_identity = json.loads(
        (run_root / "results" / "candidate-identity.json").read_text()
    )
    if candidate_identity["torch"] != build_environment["torch"]:
        raise RuntimeError("candidate runtime Torch identity differs from build-time Torch")
    baseline_prefix = site_packages(baseline_python, run_root)
    run([str(baseline_python), "-s", str(CONTEST / "scripts" / "identity.py"), "--expected-prefix", str(baseline_prefix), "--expected-torch-version", FROZEN_TORCH_VERSION, "--expected-torch-root", str(FROZEN_TORCH_ROOT), "--output", str(run_root / "results" / "baseline-identity.json")], cwd=run_root, log=run_root / "logs" / "06-baseline-identity.log", env=clean_env(baseline_python))
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
    identity = source_identity()
    status_path = run_root / "results" / "BUILD_STATUS.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "PASS" or not same_source(status.get("source", {}), identity):
        raise RuntimeError("test source identity does not match successful build")
    python = Path(status["candidate_python"])
    baseline_python = Path(args.baseline_python or config["baseline_python"])
    smoke_status = run_root / "results" / "SMOKE_STATUS.json"
    smoke_status.unlink(missing_ok=True)
    runtime_status = runtime_identity_gate(
        args, status, output_name="RUNTIME_IDENTITY_STATUS.json", log_prefix="pretest"
    )
    runtime_status_path = run_root / "results" / "RUNTIME_IDENTITY_STATUS.json"
    if runtime_status["status"] != "PASS":
        raise RuntimeError(runtime_status["status"])
    runtime_status["workers_started"] = True
    runtime_status_path.write_text(json.dumps(runtime_status, indent=2) + "\n")
    baseline_out = run_root / "results" / "smoke-baseline"
    candidate_out = run_root / "results" / "smoke-candidate"
    run([str(baseline_python), "-s", str(CONTEST / "scripts" / "worker.py"), "--mode", "baseline", "--model", str(model), "--structure", str(structure), "--output", str(baseline_out)], cwd=run_root, log=run_root / "logs" / "08-smoke-baseline.log", env=clean_env(baseline_python))
    run([str(python), "-s", str(CONTEST / "scripts" / "worker.py"), "--mode", "candidate", "--model", str(model), "--structure", str(structure), "--output", str(candidate_out)], cwd=run_root, log=run_root / "logs" / "09-smoke-candidate.log", env=clean_env(python))
    run([str(baseline_python), "-s", str(CONTEST / "scripts" / "compare.py"), "--baseline", str(baseline_out / "efs.npz"), "--candidate", str(candidate_out / "efs.npz"), "--config", str(CONFIG), "--output", str(smoke_status)], cwd=run_root, log=run_root / "logs" / "10-compare.log", env=clean_env(baseline_python))


def benchmark(args) -> None:
    config, run_root, model, structure = resolve_inputs(args)
    current = source_identity()
    build_status = json.loads((run_root / "results" / "BUILD_STATUS.json").read_text())
    smoke_status = json.loads((run_root / "results" / "SMOKE_STATUS.json").read_text())
    if build_status.get("status") != "PASS" or not same_source(build_status.get("source", {}), current):
        raise RuntimeError("benchmark source identity does not match successful build")
    if smoke_status.get("status") != "PASS":
        raise RuntimeError("benchmark requires a passing public smoke test")
    pre = runtime_identity_gate(
        args, build_status, output_name="BENCHMARK_RUNTIME_IDENTITY_PRE.json",
        log_prefix="benchmark-pre",
    )
    if pre["status"] != "PASS":
        raise RuntimeError(pre["status"])
    benchmark_root = run_root / "results" / "public-benchmark"
    if benchmark_root.exists():
        raise RuntimeError(f"benchmark output already exists: {benchmark_root}")
    baseline_python = Path(args.baseline_python or config["baseline_python"])
    candidate = Path(build_status["candidate_python"])
    run(
        [str(baseline_python), "-s", str(BENCHMARK_RUNNER),
         "--output-root", str(benchmark_root), "--baseline-python", str(baseline_python),
         "--candidate-python", str(candidate), "--worker", str(BENCHMARK_WORKER),
         "--model", str(model), "--structure", str(structure),
         "--benchmark-tolerance", str(BENCHMARK_TOLERANCE)],
        cwd=run_root, log=run_root / "logs" / "benchmark.log", env=clean_env(baseline_python),
    )
    benchmark_status = json.loads((benchmark_root / "BENCHMARK_STATUS.json").read_text())
    if benchmark_status.get("status") != "PASS":
        raise RuntimeError(str(benchmark_status.get("status")))
    post = runtime_identity_gate(
        args, build_status, output_name="BENCHMARK_RUNTIME_IDENTITY_POST.json",
        log_prefix="benchmark-post",
    )
    if post["status"] != "PASS":
        raise RuntimeError(post["status"])
    after = source_identity()
    if not same_source(current, after):
        raise RuntimeError("candidate source commit/tree changed during benchmark")
    aggregate = json.loads((benchmark_root / "result.json").read_text())
    pair_manifests = {}
    for pair_dir in sorted(benchmark_root.glob("Pair*")):
        pair_manifests[pair_dir.name] = {
            "input_manifest": sha256(pair_dir / "input-manifest.json"),
            "reference_manifest": sha256(pair_dir / "reference-manifest.json"),
        }
    starter_commit = git(["rev-parse", f"{args.starter_ref}^{{commit}}"])
    binding = {
        "schema_version": "dpa4c-ppu-contest.measurement-binding.v1",
        "status": "PASS", "score_type": "public_self_test", "verified": False,
        "starter": {"ref": args.starter_ref, "resolved_commit": starter_commit},
        "source": after,
        "tracked_candidate_manifest": build_status["tracked_candidate_manifest"],
        "build": {
            "command": build_status["build_command"],
            "configuration": build_status["build_configuration"],
            "candidate_prefix": build_status["candidate_prefix"],
            "candidate_python": build_status["candidate_python"],
            "wheel": build_status["wheel"],
        },
        "runtime": {
            "candidate": post["candidate"]["identity"],
            "baseline": post["baseline"]["identity"],
        },
        "tools": {
            "runner": {"path": str(BENCHMARK_RUNNER), "sha256": sha256(BENCHMARK_RUNNER)},
            "worker": {"path": str(BENCHMARK_WORKER), "sha256": sha256(BENCHMARK_WORKER)},
            "validator": {"path": str(BENCHMARK_RUNNER), "sha256": sha256(BENCHMARK_RUNNER)},
            "smoke_validator": {"path": str(CONTEST / "scripts" / "compare.py"),
                                "sha256": sha256(CONTEST / "scripts" / "compare.py")},
        },
        "assets": build_status["assets"],
        "benchmark_tolerance": benchmark_tolerance_identity(),
        "protocol": {"version": "dpa4c-ppu-contest.public-benchmark.v1",
                     "warmup": 20, "measured": 500, "pairs": 3,
                     "order": ["AB", "BA", "AB"]},
        "pair_manifests": pair_manifests,
        "result": {"path": str(benchmark_root / "result.json"),
                   "sha256": sha256(benchmark_root / "result.json"),
                   "paired_median_speedup": aggregate["paired_median_speedup"]},
        "formal_performance": "NOT_RUN_BY_SCOPE",
    }
    binding["output_artifacts"] = {
        str(path.relative_to(benchmark_root)): sha256(path)
        for path in sorted(benchmark_root.rglob("*")) if path.is_file()
    }
    (benchmark_root / "measurement-binding.json").write_text(json.dumps(binding, indent=2) + "\n")


def changed_files(repo: Path, starter: str, candidate: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only", "-z", starter, candidate]
    )
    return [name.decode() for name in output.split(b"\0") if name]


def reject_binary_implementation_files(paths: list[str]) -> None:
    rejected = [path for path in paths if Path(path).suffix.lower() in BINARY_IMPLEMENTATION_SUFFIXES]
    if rejected:
        raise RuntimeError(f"binary implementation artifacts are prohibited: {rejected}")


def make_patch(repo: Path, starter: str, candidate: str, output: Path) -> dict[str, object]:
    files = changed_files(repo, starter, candidate)
    reject_binary_implementation_files(files)
    patch = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "--full-index", starter, candidate]
    )
    if not patch:
        raise RuntimeError("candidate patch is empty")
    output.write_bytes(patch)
    with tempfile.TemporaryDirectory(prefix="dpa4c-patch-check-") as directory:
        checkout = Path(directory) / "checkout"
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", starter], check=True)
        subprocess.run(["git", "-C", str(checkout), "apply", "--binary", "--index", str(output)], check=True)
        reconstructed_tree = subprocess.check_output(
            ["git", "-C", str(checkout), "write-tree"], text=True
        ).strip()
    candidate_tree = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{candidate}^{{tree}}"], text=True
    ).strip()
    if reconstructed_tree != candidate_tree:
        raise RuntimeError(f"patch reconstructs {reconstructed_tree}, expected {candidate_tree}")
    return {"changed_files": files, "sha256": sha256(output),
            "reconstructed_tree": reconstructed_tree}


def package(args) -> None:
    _, run_root, _, _ = resolve_inputs(args)
    current = source_identity()
    benchmark_root = run_root / "results" / "public-benchmark"
    benchmark_status = json.loads((benchmark_root / "BENCHMARK_STATUS.json").read_text())
    binding = json.loads((benchmark_root / "measurement-binding.json").read_text())
    starter = git(["rev-parse", f"{args.starter_ref}^{{commit}}"])
    validate_package_gate(benchmark_status, binding, current, args.starter_ref, starter)
    output = run_root / "submission"
    if output.exists():
        raise RuntimeError(f"submission output already exists: {output}")
    output.mkdir(parents=True)
    patch_info = make_patch(ROOT, starter, current["commit"], output / "candidate.patch")
    for name in ("result.json", "repeats.json", "measurement-binding.json"):
        shutil.copy2(benchmark_root / name, output / name)
    image = {"schema_version": "dpa4c-ppu-contest.image.v1", "status": "NOT_RUN",
             "reason": "image is outside the public source submission flow"}
    (output / "image.json").write_text(json.dumps(image, indent=2) + "\n")
    changelog = git(["log", "--format=%h %s", f"{starter}..{current['commit']}"])
    (output / "CHANGELOG.md").write_text(
        "# Candidate changes\n\n" + (changelog or "No commit messages.\n") + "\n"
    )
    manifest = {
        "schema_version": "dpa4c-ppu-contest.submission.v1", "status": "PASS",
        "starter": {"ref": args.starter_ref, "resolved_commit": starter},
        "candidate": current, "patch": patch_info,
        "score_type": "public_self_test", "verified": False,
        "files": {}, "formal_performance": "NOT_RUN_BY_SCOPE",
    }
    for path in sorted(output.iterdir()):
        if path.name not in {"submission-manifest.json", "SHA256SUMS"}:
            manifest["files"][path.name] = sha256(path)
    (output / "submission-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    payloads = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in payloads)
    )
    expected = {"candidate.patch", "result.json", "repeats.json",
                "measurement-binding.json", "submission-manifest.json", "image.json",
                "CHANGELOG.md", "SHA256SUMS"}
    if {path.name for path in output.iterdir()} != expected:
        raise RuntimeError("submission output set differs from fixed contract")


def image_status(args) -> None:
    _, run_root, _, _ = resolve_inputs(args)
    target = run_root / "results" / "IMAGE_STATUS.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "schema_version": "dpa4c-ppu-contest.image-command.v1", "status": "NOT_RUN",
        "reason": "controlled placeholder; this tool does not call Bohrium or build images",
    }, indent=2) + "\n")


def validate_package_gate(
    benchmark_status: dict[str, object], binding: dict[str, object],
    current: dict[str, str], starter_ref: str, starter_commit: str,
) -> None:
    if benchmark_status.get("status") != "PASS" or binding.get("status") != "PASS":
        raise RuntimeError("package requires a passing public benchmark and binding")
    if not same_source(binding.get("source", {}), current):
        raise RuntimeError("measured commit/tree differs from current candidate")
    if binding.get("starter") != {"ref": starter_ref, "resolved_commit": starter_commit}:
        raise RuntimeError("starter ref differs from measured binding")


def execute_command(args) -> None:
    if args.command == "build":
        build(args)
    elif args.command == "test":
        test(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "package":
        package(args)
    elif args.command == "image":
        image_status(args)
    elif args.command == "all":
        build(args)
        test(args)
        benchmark(args)
        package(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="DPA4C PPU public contestant source flow")
    parser.add_argument("command", choices=("build", "test", "benchmark", "package", "all", "image"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--baseline-python")
    parser.add_argument("--starter-ref", default=DEFAULT_STARTER_REF,
                        help="explicit immutable starter tag/commit used by benchmark binding and package")
    args = parser.parse_args()
    try:
        execute_command(args)
        print(json.dumps({"status": "PASS", "command": args.command, "run_root": str(args.run_root.resolve()), "formal_performance": "NOT_RUN_BY_SCOPE"}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "command": args.command, "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
