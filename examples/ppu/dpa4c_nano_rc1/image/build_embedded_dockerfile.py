#!/usr/bin/env python3
"""Generate a <=64 KiB Dockerfile chain embedding a verified sparse Git checkout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import lzma
import subprocess
import tarfile
import tempfile
import zlib
from pathlib import Path

CANDIDATE = "8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2"
CANDIDATE_TREE = "e4e5897d68bceeb7c05b0256bdfaa12bcf0002d2"
BASE = "3d079bdfb3d5ba8b5604bdedff7d196082f7d80b"
BASE_TREE = "1afe818872863e93edd3565b0d32ba2d79942301"
PATCH_SHA = "1bb563718ac189856fa0d40982c18b0fec12f21c5c378954ea9a74cee3309611"
TAG = "dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1"
PUBLIC_URL = "https://github.com/yangchaoss/deepmd-kit.git"
SPARSE_PATH = "examples/ppu/dpa4c_nano_rc1"
RUNTIME_IMAGE = (
    "registry.dp.tech/dptech/dp/native/prod-27666/4553069/"
    "dpa4c-nano-rc1-runtime-probe:20260911-yangchao-r1@"
    "sha256:f475eb7a0488b8785818b506fc17e9c411b7ef526fdae2d974c62a5c42937061"
)


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_path(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    tar.add(path, arcname=arcname, recursive=True)


def write_xz_tar(path: Path, entries: list[tuple[Path, str]]) -> None:
    with lzma.open(path, "wb", preset=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for source, arcname in entries:
                add_path(tar, source, arcname)


def encode_run(archive: Path, target: str) -> str:
    encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    chunks = " \\\n".join(f"    '{encoded[i:i + 4096]}'" for i in range(0, len(encoded), 4096))
    return f"""RUN set -eux; \\
    mkdir -p {target}; \\
    printf '%s' \\
{chunks} \\
      | base64 -d > /tmp/payload.tar.xz; \\
    echo '{sha256(archive)}  /tmp/payload.tar.xz' | sha256sum -c -; \\
    tar -xJf /tmp/payload.tar.xz -C {target}; \\
    rm /tmp/payload.tar.xz
"""


def encode_chunk(data: bytes, target: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    chunks = " \\\n".join(f"    '{encoded[i:i + 4096]}'" for i in range(0, len(encoded), 4096))
    return f"""RUN set -eux; \\
    mkdir -p /tmp/dpa4c-git-objects; \\
    printf '%s' \\
{chunks} \\
      | base64 -d > {target}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image-prefix", required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()

    assert output("git", "-C", str(repo), "rev-parse", CANDIDATE) == CANDIDATE
    assert output("git", "-C", str(repo), "rev-parse", f"{CANDIDATE}^{{tree}}") == CANDIDATE_TREE
    assert output("git", "-C", str(repo), "rev-parse", f"{BASE}^{{tree}}") == BASE_TREE

    with tempfile.TemporaryDirectory(prefix="dpa4c-split-git-") as temp:
        root = Path(temp)
        checkout = root / "repo"
        run(
            "git", "clone", "--no-local", "--depth", "1", "--branch", TAG,
            "--single-branch", f"file://{repo}", str(checkout),
        )
        run("git", "-C", str(checkout), "fetch", "--depth", "1", f"file://{repo}", BASE)
        run("git", "-C", str(checkout), "checkout", "--detach", CANDIDATE)
        run("git", "-C", str(checkout), "sparse-checkout", "init", "--cone", "--sparse-index")
        run("git", "-C", str(checkout), "sparse-checkout", "set", SPARSE_PATH)
        run("git", "-C", str(checkout), "config", "core.abbrev", "8")

        assert not output("git", "-C", str(checkout), "status", "--porcelain")
        actual_patch = subprocess.check_output(
            ["git", "-C", str(checkout), "diff", "--binary", "--no-ext-diff", "--no-textconv", BASE, CANDIDATE, "--"]
        )
        assert hashlib.sha256(actual_patch).hexdigest() == PATCH_SHA
        run("git", "-C", str(checkout), "sparse-checkout", "reapply", "--sparse-index")

        object_ids = set(output("git", "-C", str(repo), "rev-list", "--ancestry-path", f"{BASE}..{CANDIDATE}").splitlines())
        object_ids.add(BASE)
        for commit in (BASE, CANDIDATE):
            object_ids.add(output("git", "-C", str(repo), "rev-parse", f"{commit}^{{tree}}"))
            for spec in (commit, f"{commit}:examples", f"{commit}:examples/ppu"):
                for line in output("git", "-C", str(repo), "ls-tree", spec).splitlines():
                    fields = line.split()
                    if fields[1] == "tree":
                        object_ids.add(fields[2])
            for line in output("git", "-C", str(repo), "ls-tree", "-r", "-t", commit, "--", SPARSE_PATH).splitlines():
                object_ids.add(line.split()[2])

        object_list = root / "objects.txt"
        object_list.write_text("\n".join(sorted(object_ids)) + "\n")
        minimal_pack = root / "minimal.pack"
        with object_list.open("rb") as source, minimal_pack.open("wb") as target:
            subprocess.run(["git", "-C", str(repo), "pack-objects", "--stdout"], stdin=source, stdout=target, check=True)

        pack_dir = checkout / ".git/objects/pack"
        for item in pack_dir.iterdir():
            item.unlink()
        with minimal_pack.open("rb") as source:
            subprocess.run(["git", "-C", str(checkout), "index-pack", "--stdin", "--fix-thin"], stdin=source, check=True, stdout=subprocess.DEVNULL)
        (checkout / ".git/shallow").write_text(BASE + "\n")
        run("git", "-C", str(checkout), "remote", "set-url", "origin", PUBLIC_URL)

        objects_archive = root / "objects.tar.xz"
        write_xz_tar(
            objects_archive,
            [(item, f".git/objects/pack/{item.name}") for item in sorted(pack_dir.iterdir())],
        )

        support_root = root / "support"
        root_entries: list[tuple[Path, str]] = []
        for item in sorted(checkout.iterdir()):
            if item.name == ".git" or not (item.is_file() or item.is_symlink()):
                continue
            root_entries.append((item, item.name))
            blob = subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"{CANDIDATE}:{item.name}"])
            loose = b"blob " + str(len(blob)).encode("ascii") + b"\0" + blob
            object_id = hashlib.sha1(loose).hexdigest()
            assert object_id == output("git", "-C", str(repo), "rev-parse", f"{CANDIDATE}:{item.name}")
            loose_path = support_root / ".git/objects" / object_id[:2] / object_id[2:]
            loose_path.parent.mkdir(parents=True, exist_ok=True)
            loose_path.write_bytes(zlib.compress(loose, level=9))
            root_entries.append((loose_path, f".git/objects/{object_id[:2]}/{object_id[2:]}"))
        support_archive = root / "support.tar.xz"
        write_xz_tar(support_archive, root_entries)

        worktree_archive = root / "worktree.tar.xz"
        metadata = (
            ".git/HEAD",
            ".git/config",
            ".git/config.worktree",
            ".git/index",
            ".git/shallow",
            ".git/info/sparse-checkout",
            ".git/objects/info",
            ".git/refs",
        )
        write_xz_tar(
            worktree_archive,
            [(checkout / relative, relative) for relative in metadata]
            + [(checkout / SPARSE_PATH, SPARSE_PATH)],
        )

        reconstructed = root / "reconstructed"
        reconstructed.mkdir()
        with tarfile.open(objects_archive, "r:xz") as tar:
            tar.extractall(reconstructed)
        with tarfile.open(support_archive, "r:xz") as tar:
            tar.extractall(reconstructed)
        with tarfile.open(worktree_archive, "r:xz") as tar:
            tar.extractall(reconstructed)
        assert output("git", "-C", str(reconstructed), "rev-parse", "HEAD") == CANDIDATE
        assert output("git", "-C", str(reconstructed), "rev-parse", "HEAD^{tree}") == CANDIDATE_TREE
        assert output("git", "-C", str(reconstructed), "rev-parse", f"{BASE}^{{tree}}") == BASE_TREE
        assert not output("git", "-C", str(reconstructed), "status", "--porcelain")
        reconstructed_patch = subprocess.check_output(
            ["git", "-C", str(reconstructed), "diff", "--binary", "--no-ext-diff", "--no-textconv", BASE, CANDIDATE, "--"]
        )
        assert hashlib.sha256(reconstructed_patch).hexdigest() == PATCH_SHA

        args.output_dir.mkdir(parents=True, exist_ok=True)
        object_bytes = objects_archive.read_bytes()
        chunk_size = 40000
        object_chunks = [object_bytes[i:i + chunk_size] for i in range(0, len(object_bytes), chunk_size)]
        generated: list[Path] = []
        previous_image = RUNTIME_IMAGE
        for index, chunk in enumerate(object_chunks, start=1):
            image = f"{args.image_prefix}:{args.tag_prefix}-objects-{index:02d}"
            dockerfile = f"FROM {previous_image}\n\n" + encode_chunk(
                chunk, f"/tmp/dpa4c-git-objects/part-{index:02d}"
            )
            if index == len(object_chunks):
                parts = " ".join(
                    f"/tmp/dpa4c-git-objects/part-{part:02d}"
                    for part in range(1, len(object_chunks) + 1)
                )
                dockerfile += f"""
RUN set -eux; \\
    mkdir -p /opt/dpa4c-contestant-kit; \\
    cat {parts} > /tmp/objects.tar.xz; \\
    echo '{sha256(objects_archive)}  /tmp/objects.tar.xz' | sha256sum -c -; \\
    tar -xJf /tmp/objects.tar.xz -C /opt/dpa4c-contestant-kit; \\
    rm -rf /tmp/dpa4c-git-objects /tmp/objects.tar.xz
"""
            dockerfile += "\nWORKDIR /workspace\n"
            path = args.output_dir / f"Dockerfile.objects-{index:02d}"
            path.write_text(dockerfile)
            generated.append(path)
            previous_image = image

        last_object_image = previous_image
        support_bytes = support_archive.read_bytes()
        support_chunks = [support_bytes[i:i + chunk_size] for i in range(0, len(support_bytes), chunk_size)]
        for index, chunk in enumerate(support_chunks, start=1):
            image = f"{args.image_prefix}:{args.tag_prefix}-support-{index:02d}"
            dockerfile = f"FROM {previous_image}\n\n" + encode_chunk(
                chunk, f"/tmp/dpa4c-git-support/part-{index:02d}"
            ).replace("mkdir -p /tmp/dpa4c-git-objects", "mkdir -p /tmp/dpa4c-git-support")
            if index == len(support_chunks):
                parts = " ".join(
                    f"/tmp/dpa4c-git-support/part-{part:02d}"
                    for part in range(1, len(support_chunks) + 1)
                )
                dockerfile += f"""
RUN set -eux; \\
    cat {parts} > /tmp/support.tar.xz; \\
    echo '{sha256(support_archive)}  /tmp/support.tar.xz' | sha256sum -c -; \\
    tar -xJf /tmp/support.tar.xz -C /opt/dpa4c-contestant-kit; \\
    rm -rf /tmp/dpa4c-git-support /tmp/support.tar.xz
"""
            dockerfile += "\nWORKDIR /workspace\n"
            path = args.output_dir / f"Dockerfile.support-{index:02d}"
            path.write_text(dockerfile)
            generated.append(path)
            previous_image = image

        final = f"FROM {previous_image}\n\n" + encode_run(worktree_archive, "/opt/dpa4c-contestant-kit") + f"""
RUN set -eux; \\
    test \"$(git -C /opt/dpa4c-contestant-kit rev-parse HEAD)\" = {CANDIDATE}; \\
    test \"$(git -C /opt/dpa4c-contestant-kit rev-parse HEAD^{{tree}})\" = {CANDIDATE_TREE}; \\
    test \"$(git -C /opt/dpa4c-contestant-kit rev-parse {BASE}^{{tree}})\" = {BASE_TREE}; \\
    test \"$(git -C /opt/dpa4c-contestant-kit diff --binary --no-ext-diff --no-textconv {BASE} HEAD -- | sha256sum | awk '{{print $1}}')\" = {PATCH_SHA}; \\
    test -z \"$(git -C /opt/dpa4c-contestant-kit status --porcelain)\"

ENV DPA4C_CONTESTANT_ROOT=/opt/dpa4c-contestant-kit

LABEL org.opencontainers.image.source=\"{PUBLIC_URL}\" \\
      com.dptech.dpa4c.candidate-commit=\"{CANDIDATE}\" \\
      com.dptech.dpa4c.candidate-tree=\"{CANDIDATE_TREE}\" \\
      com.dptech.dpa4c.patch-sha256=\"{PATCH_SHA}\"

WORKDIR /workspace
"""
        final_path = args.output_dir / "Dockerfile.final"
        final_path.write_text(final)
        generated.append(final_path)
        for path in generated:
            if path.stat().st_size > 65536:
                raise SystemExit(f"{path} is {path.stat().st_size} bytes; exceeds Bohrium 65536-byte limit")
            print(f"{path.name}_BYTES={path.stat().st_size}")
            print(f"{path.name}_SHA256={sha256(path)}")
        print(f"OBJECT_STAGE_COUNT={len(object_chunks)}")
        print(f"LAST_OBJECT_IMAGE={last_object_image}")
        print(f"SUPPORT_STAGE_COUNT={len(support_chunks)}")
        print(f"LAST_SUPPORT_IMAGE={previous_image}")
        print(f"OBJECTS_ARCHIVE_SHA256={sha256(objects_archive)}")
        print(f"SUPPORT_ARCHIVE_SHA256={sha256(support_archive)}")
        print(f"WORKTREE_ARCHIVE_SHA256={sha256(worktree_archive)}")


if __name__ == "__main__":
    main()
