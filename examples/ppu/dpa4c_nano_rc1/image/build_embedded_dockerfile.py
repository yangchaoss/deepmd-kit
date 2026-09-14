#!/usr/bin/env python3
"""Generate a context-free Bohrium Dockerfile containing a real shallow Git checkout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

CANDIDATE = "8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2"
CANDIDATE_TREE = "e4e5897d68bceeb7c05b0256bdfaa12bcf0002d2"
BASE = "3d079bdfb3d5ba8b5604bdedff7d196082f7d80b"
BASE_TREE = "1afe818872863e93edd3565b0d32ba2d79942301"
TAG = "dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1"
PUBLIC_URL = "https://github.com/yangchaoss/deepmd-kit.git"
SPARSE_PATH = "examples/ppu/dpa4c_nano_rc1"


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime-dockerfile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    assert run("git", "-C", str(repo), "rev-parse", CANDIDATE) == CANDIDATE
    assert run("git", "-C", str(repo), "rev-parse", f"{CANDIDATE}^{{tree}}") == CANDIDATE_TREE
    assert run("git", "-C", str(repo), "rev-parse", f"{BASE}^{{tree}}") == BASE_TREE

    with tempfile.TemporaryDirectory(prefix="dpa4c-embedded-git-") as temp:
        root = Path(temp)
        checkout = root / "repo"
        subprocess.run(
            [
                "git", "clone", "--no-local", "--depth", "1", "--branch", TAG,
                "--single-branch", f"file://{repo}", str(checkout),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", SPARSE_PATH], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth", "1", f"file://{repo}", BASE],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", CANDIDATE], check=True)
        subprocess.run(["git", "-C", str(checkout), "remote", "set-url", "origin", PUBLIC_URL], check=True)
        subprocess.run(["git", "-C", str(checkout), "config", "core.abbrev", "8"], check=True)
        assert run("git", "-C", str(checkout), "rev-parse", "HEAD") == CANDIDATE
        assert run("git", "-C", str(checkout), "rev-parse", "HEAD^{tree}") == CANDIDATE_TREE
        assert run("git", "-C", str(checkout), "rev-parse", f"{BASE}^{{tree}}") == BASE_TREE
        assert not run("git", "-C", str(checkout), "status", "--porcelain")

        archive = root / "dpa4c-contestant-git.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=9) as tar:
            tar.add(checkout, arcname="dpa4c-contestant-kit", recursive=True)
        archive_sha = sha256(archive)
        encoded = base64.b64encode(archive.read_bytes()).decode("ascii")

    runtime = args.runtime_dockerfile.read_text()
    marker = "\nWORKDIR /workspace\n"
    if marker not in runtime:
        raise SystemExit("runtime Dockerfile has no final WORKDIR marker")
    runtime = runtime.replace(marker, "\n", 1)
    chunks = " \\\n".join(f"    '{encoded[i:i + 4096]}'" for i in range(0, len(encoded), 4096))
    source_layer = f"""
RUN set -eux; \\
    printf '%s' \\
{chunks} \\
      | base64 -d > /tmp/dpa4c-contestant-git.tar.gz; \\
    echo '{archive_sha}  /tmp/dpa4c-contestant-git.tar.gz' | sha256sum -c -; \\
    mkdir -p /opt; \\
    tar -xzf /tmp/dpa4c-contestant-git.tar.gz -C /opt; \\
    rm /tmp/dpa4c-contestant-git.tar.gz; \\
    test \"$(git -C /opt/dpa4c-contestant-kit rev-parse HEAD)\" = {CANDIDATE}; \\
    test \"$(git -C /opt/dpa4c-contestant-kit rev-parse HEAD^{{tree}})\" = {CANDIDATE_TREE}; \\
    test \"$(git -C /opt/dpa4c-contestant-kit rev-parse {BASE}^{{tree}})\" = {BASE_TREE}; \\
    test -z \"$(git -C /opt/dpa4c-contestant-kit status --porcelain)\"

ENV DPA4C_CONTESTANT_ROOT=/opt/dpa4c-contestant-kit

LABEL com.dptech.dpa4c.candidate-commit=\"{CANDIDATE}\" \\
      com.dptech.dpa4c.candidate-tree=\"{CANDIDATE_TREE}\" \\
      com.dptech.dpa4c.embedded-git-archive-sha256=\"{archive_sha}\"

WORKDIR /workspace
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(runtime + source_layer)
    print(f"OUTPUT={args.output}")
    print(f"ARCHIVE_SHA256={archive_sha}")
    print(f"DOCKERFILE_SHA256={sha256(args.output)}")
    print(f"DOCKERFILE_BYTES={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
