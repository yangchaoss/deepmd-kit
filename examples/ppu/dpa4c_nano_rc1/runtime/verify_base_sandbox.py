#!/usr/bin/env python3
"""Verify the frozen PPU runtime image in a fresh device Sandbox."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import torch


EXPECTED_TORCH = "2.9.0+ali.10.ppu2.1.0.cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_DEVICE_COUNT = 1


def command_path(name: str) -> str | None:
    result = subprocess.run(
        ["bash", "-lc", f"command -v {name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def main() -> None:
    checks = {
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "nvcc_path": command_path("nvcc"),
        "ppu_smi_path": command_path("ppu-smi"),
        "ppu_sdk": os.environ.get("PPU_SDK"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "python": platform.python_version(),
    }

    assert checks["torch_version"] == EXPECTED_TORCH, checks
    assert checks["torch_cuda_version"] == EXPECTED_CUDA, checks
    assert checks["cuda_available"] is True, checks
    assert checks["device_count"] == EXPECTED_DEVICE_COUNT, checks
    assert checks["nvcc_path"], checks
    assert checks["ppu_smi_path"], checks
    assert Path(str(checks["nvcc_path"])).is_file(), checks

    device = torch.device("cuda:0")
    lhs = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    rhs = torch.eye(4, dtype=torch.float32, device=device)
    actual = lhs @ rhs
    torch.cuda.synchronize(device)
    assert torch.equal(actual.cpu(), lhs.cpu())

    checks["device_name"] = torch.cuda.get_device_name(0)
    checks["tensor_smoke"] = "PASS"
    checks["status"] = "PASS"
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
