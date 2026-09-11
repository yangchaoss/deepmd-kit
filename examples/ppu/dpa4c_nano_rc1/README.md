# DPA4C PPU Nano Golden Prototype v1.0-rc1

This directory freezes the source side of the private PPU Nano evaluation
prototype. It is an evaluation-infrastructure calibration payload, not a
restriction on the implementation boundaries of a future migration contest.

## Frozen identities

- Upstream repository: `https://github.com/deepmodeling/deepmd-kit.git`
- Upstream source commit: `14a71f13bb840c10d78465f75a00e3a31764a0fd`
- Backend exercised by the verified model: `deepmd.pt_expt`
- Model asset: DPA4C Nano checkpoint, external and SHA256-bound
- Structure asset: fully periodic 1024-atom structure, external and
  SHA256-bound
- Precision and outer execution contract: FP32, eager, single PPU

The model, private input sequence, credentials, PPU SDK, compiled objects and
performance evidence are deliberately not stored in Git. See
`source-lock.json` for the verified external asset hashes.

## What is frozen here

The candidate CUDA source performs the live edge-gradient to force,
atom-virial and frame-virial assembly used by the DPA4C E/F/S path. The Python
adapter installs that operator at the current `pt_expt` seam. The tracked
build, probe and E/F/S scripts reproduce the already demonstrated development
path without relying on untracked implementation files.

The force/virial seam is only a Golden Candidate payload. A future contestant
may optimize or replace any part of the DPA4C GPU execution path while keeping
the external model and E/F/S contract.

## Development replay

Run only inside the approved PPU runtime environment:

```bash
export PPU_SDK=/usr/local/PPU_SDK
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export DPA4C_RC1_MODEL=/external/DPA4C-Nano.pt
export DPA4C_RC1_STRUCTURE=/external/structure-1024.extxyz
export DPA4C_RC1_WORK_ROOT=/external/work/dpa4c-nano-rc1
bash examples/ppu/dpa4c_nano_rc1/run_development_probe.sh
```

`DPA4C_RC1_WORK_ROOT` must be outside the Git checkout. Build products and
evidence are written below it, keeping the controlled source tree clean.
This entry point remains a development probe; it does not run the formal
20-warmup + 500-measured paired benchmark.

## Remaining environment gate

The source baseline is frozen by this branch. The Frozen PPU Runtime Image is
a separate identity and is not yet declared here because no verified Bohrium
Image ID/digest and successful build log are currently available. Do not infer
runtime-image reproducibility from this source freeze.

The candidate-independent recipe for the first infrastructure gate is tracked
at `runtime/Dockerfile.minimal-probe`. It verifies the base image, PPU SDK CUDA
compiler wrapper and frozen PyTorch/CUDA identity during image construction.
Passing that build is necessary but not sufficient: a fresh PPU Sandbox must
still boot from the resulting image and pass device/runtime smoke checks before
the image identity can be frozen in `source-lock.json`.
