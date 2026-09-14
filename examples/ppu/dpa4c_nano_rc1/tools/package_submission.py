#!/usr/bin/env python3
"""Create the candidate.patch + result + manifest submission contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"submission output already exists: {output}")
    if git(repo, "status", "--porcelain=v1"):
        raise SystemExit("candidate repository must be clean and committed")
    head = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    git(repo, "cat-file", "-e", args.base_commit + "^{commit}")
    output.mkdir(parents=True)
    patch = output / "candidate.patch"
    patch.write_bytes(subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "--full-index", "--no-ext-diff", args.base_commit, head, "--"]))
    if not patch.stat().st_size:
        raise SystemExit("candidate patch is empty")
    shutil.copy2(args.smoke, output / "smoke.json")
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    raw = subprocess.check_output(["git", "-C", str(repo), "diff", "--name-status", "--no-renames", "-z", args.base_commit, head, "--"])
    tokens = [item.decode() for item in raw.split(b"\0") if item]
    changed = []
    for index in range(0, len(tokens), 2):
        status, relative = tokens[index][0], tokens[index + 1]
        target = repo / relative
        changed.append({"path": relative, "status": status, "sha256": None if status == "D" else sha256(target)})
    rc1 = Path("examples/ppu/dpa4c_nano_rc1")
    script_paths = {"build": rc1 / "scripts/build.sh", "install": rc1 / "scripts/install.sh", "run": rc1 / "scripts/run.sh"}
    scripts = {
        name: {"path": str(path), "sha256": sha256(repo / path), "command": str(path)}
        for name, path in script_paths.items()
    }
    remote = git(repo, "remote", "get-url", "origin")
    note = rc1 / "CUDA_CHANGE.md"
    result = json.loads(args.result.read_text(encoding="utf-8"))
    result["submission_identity"] = {
        "repository_url": remote,
        "immutable_commit": head,
        "tree": tree,
        "base_commit": args.base_commit,
        "patch_sha256": sha256(patch),
        "model_sha256": smoke.get("model_sha256"),
        "structure_sha256": smoke.get("structure_sha256"),
        "protocol": smoke.get("schema_version"),
        "actual_route": smoke.get("adapter", {}).get("dispatch"),
        "shared_object_sha256": smoke.get("shared_object_sha256"),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "dpa4c-contest-runtime.submission-manifest.v2",
        "status": "READY",
        "candidate": {
            "repository_url": remote,
            "immutable_commit": head,
            "tree": tree,
            "patch": {"path": "candidate.patch", "base_commit": args.base_commit, "sha256": sha256(patch)},
            "files": changed,
            "scripts": scripts,
            "entrypoint": {
                "kind": "custom_extension",
                "install_module": "runner.runtime_adapter",
                "install_symbol": "install",
                "calculator_module": "deepmd.calculator",
                "calculator_symbol": "DP"
            },
            "cuda_change_note": {"path": str(note), "sha256": sha256(repo / note)}
        },
        "self_test_result": {"path": "result.json", "sha256": sha256(output / "result.json")},
        "public_smoke": {"path": "smoke.json", "sha256": sha256(output / "smoke.json")},
        "formal_policy": "Organizer-controlled rebuild and private validation determine the final score."
    }
    manifest_path = output / "submission-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS":
            checksum_lines.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "commit": head, "tree": tree, "patch_sha256": sha256(patch), "files": sorted(path.name for path in output.iterdir())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
