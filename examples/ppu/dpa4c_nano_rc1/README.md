# DPA4C PPU Nano contestant kit

This tracked kit is the public, reproducible entrypoint for the DPA4C Nano
CUDA/PPU force-and-virial candidate. A contestant clones the repository at a
fixed commit, supplies the external model and 1024-atom structure, and writes
all environments, build products, results and submission files outside Git.

## Environment

The supported runtime is the Bohrium PPU template
`ppu-training-xpu-2604-0908` with:

- `/opt/ac2/bin/python` and its installed `deepmd`, `deepmd.lib`, ASE and Torch;
- `/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc`;
- one visible PPU device;
- model SHA256 `f894ac16adfb7f5030d4fe4e2849db6c608f9c7074badfafb4a50c9b9afed00a`;
- structure SHA256 `137056e51cf63bd7dabf0508a29959218baf109da0c5e19a3765d11873c891e7`.

The model and structure remain external. Credentials, private seeds,
reference outputs, compiled objects and prior results are never stored here.

## Clone and run

```bash
git clone https://github.com/yangchaoss/deepmd-kit.git
cd deepmd-kit
git checkout dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1

examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh all \
  --model /external/DPA4C-Nano-OMat24-v20260819.pt \
  --structure /external/common-structure-1024.extxyz \
  --work-root /external/dpa4c-nano-run
```

For an image/bootstrap pipeline that must create a checkout non-interactively,
use the tracked `scripts/clone_fixed.sh --repository URL --ref COMMIT
--checkout NEW_DIR`; it requires an immutable 40-hex commit and refuses an
existing target directory.

`--work-root` must be outside the checkout. The command creates one isolated
candidate venv, compiles and binds the current source, runs the device dispatch
check, executes a public `1 warmup + 2 measured` E/F/virial/stress smoke,
computes a clearly labelled non-formal speedup preview, and packages the
submission.

Individual stages are also available:

```bash
.../scripts/contestant.sh bootstrap --model MODEL --structure STRUCTURE --work-root WORK
.../scripts/contestant.sh build     --model MODEL --structure STRUCTURE --work-root WORK
.../scripts/contestant.sh check     --model MODEL --structure STRUCTURE --work-root WORK
.../scripts/contestant.sh benchmark --model MODEL --structure STRUCTURE --work-root WORK
.../scripts/contestant.sh score     --model MODEL --structure STRUCTURE --work-root WORK
.../scripts/contestant.sh package   --model MODEL --structure STRUCTURE --work-root WORK
```

No manual `PYTHONPATH`, `PYTHONHOME`, Conda activation, compiler export or old
`.so` is required. The scripts scrub inherited Python overlays and import
DeepMD from the isolated venv's installed site-packages, preventing a source
checkout from hiding `deepmd.lib`.

## Outputs

```text
WORK/
  venv/                  isolated candidate Python environment
  build/                 current build.json, install.json and .so
  logs/                  stage logs
  results/check.json     live device dispatch and numerical micro-check
  results/smoke.json     public E/F/virial/stress smoke and timing preview
  results/result.json    machine-readable public score preview
  submission/
    candidate.patch
    result.json
    smoke.json
    submission-manifest.json
    SHA256SUMS
```

The submission contract is `candidate.patch + result.json + submission-manifest.json`;
`smoke.json` and `SHA256SUMS` carry the public
evidence and transfer integrity. The manifest binds repository URL, immutable
commit/tree, exact base patch, every changed file, tracked build/install/run
scripts, CUDA note, model/input result hashes and actual route.

The default submission base is frozen commit
`3d079bdfb3d5ba8b5604bdedff7d196082f7d80b`. Override `--base-commit` only
when the organizer publishes a different frozen base.

## Scoring boundary

The public `score` command reports only
`baseline_p50 / candidate_p50` from the short smoke. It does not create an
official score. The organizer independently checks the contract, rebuilds the
candidate, records the actually loaded `.so`, validates E/F/virial/stress and
runs the private frozen paired protocol before assigning any final score.
Formal seeds, reference arrays, credentials and the formal orchestrator stay
in the separate private evaluator repository.

## Runtime image recipe

`image/Dockerfile` builds the pinned PPU SDK/CUDA/Torch runtime. Source remains
in this public repository and is fetched at the fixed rc1 tag after the sandbox
starts; this avoids coupling image construction to external Git network
availability. The image contains no source, model, structure, candidate result,
private evaluator or credential.

For a self-contained image, `image/build_embedded_dockerfile.py` generates a
context-free Dockerfile containing a real shallow Git checkout of the fixed rc1
candidate plus its frozen-base objects. It verifies the original commit and
tree identities without contacting GitHub during the Bohrium image build.

Inside that image, fetch the public source and run:

```bash
git clone --branch dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1 --depth 1 \
  https://github.com/yangchaoss/deepmd-kit.git
cd deepmd-kit
test "$(git rev-parse HEAD)" = 8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2

examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh all \
  --model /external/DPA4C-Nano-OMat24-v20260819.pt \
  --structure /external/common-structure-1024.extxyz \
  --work-root /workspace/dpa4c-nano-run
```

## Troubleshooting

- `deepmd.lib` missing: do not add the repository root to `PYTHONPATH`; rerun
  `bootstrap` with the supported `/opt/ac2` runtime.
- compiler missing: confirm the exact PPU template and
  `/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc`.
- asset hash mismatch: stop and obtain the frozen external model/structure;
  do not substitute data.
- public smoke failure: retain `WORK/logs` and `WORK/results`; do not package a
  PASS result.

`CUDA_CHANGE.md`, `source-lock.json` and
`protocol/public-smoke-tolerance.json` provide the source and numeric contract.
