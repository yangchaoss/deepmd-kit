# DPA4C PPU contestant source flow (authoritative)

This file is the only authoritative contestant-facing contract for the Nano
PPU flow. The starter repository is
`https://github.com/yangchaoss/deepmd-kit.git`; the immutable starter ref is
`dpa4c-ppu-nano-starter-v1.0.0-rc3`. Resolve that tag to record the exact
commit and tree used by a run. Historical files under
`examples/ppu/dpa4c_nano_rc1/` are compatibility material only and do not
define a second submission contract.

The published `dpa4c-ppu-nano-starter-v1.0.0-rc1` and
`dpa4c-ppu-nano-starter-v1.0.0-rc2` tags are superseded historical material
and must not be used as the current starter. rc2 was superseded after final
Runtime Image prevalidation exposed candidate dependency inheritance and
failed the isolation gate; rc1 and rc2 remain history only.

The organizer calls one tracked entrypoint.  Assets remain external and are
accepted only at the frozen SHA-256 values in `config/runtime.json`.

```bash
./contest/contest.sh build --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh test  --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh benchmark --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets --starter-ref dpa4c-ppu-nano-starter-v1.0.0-rc3
./contest/contest.sh package --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets --starter-ref dpa4c-ppu-nano-starter-v1.0.0-rc3
```

`build` creates a candidate virtual environment outside the checkout, builds a
non-editable wheel from the current committed tree, installs that wheel, then
installs the tracked, hash-locked candidate runtime requirements from
`config/runtime-requirements.txt` before any identity check. The build record
binds the requirements path and SHA-256, exact install command, and resolved ASE
version and module path inside the candidate environment. It then installs the
small candidate session contract and records baseline/candidate package and ELF
identities. Before starting either worker, `test` re-records both runtime
identities and verifies stable package, ELF, Torch, and candidate-entry bindings
against `BUILD_STATUS.json`; mismatches fail closed in
`RUNTIME_IDENTITY_STATUS.json`.  Passing identities then run baseline and
candidate in separate processes from an external CWD and apply the frozen public
Nano/1024 FP32 E/F/virial/stress tolerances.

The official runtime image provides the complete wheelhouse at
`/opt/dpa4c-contest-wheelhouse`. Both build and runtime requirements are
installed strictly offline with `--no-index --find-links`, `--no-deps`, and
`--require-hashes --ignore-installed`; the checked wheel filenames and SHA-256 values are bound by
`config/wheelhouse-manifest.json`. Organizers may override only the wheelhouse
path with `DPA4C_CONTEST_WHEELHOUSE` for controlled debugging. Contestants do
not resolve or download dependencies and only run the one tracked entrypoint.
The image recipe may populate the directory with:

```bash
python -m pip download --only-binary=:all: --no-deps --require-hashes \
  --dest /opt/dpa4c-contest-wheelhouse \
  -r contest/config/build-requirements.txt \
  -r contest/config/runtime-requirements.txt
```

`benchmark` is a public, unverified self-test. It runs three fresh-process
baseline/candidate pairs in AB/BA/AB order, with 20 warmup and 500 measured
frames per route. Each route uses a separate process; reference evaluation is
also separate. Host inputs are materialized before timing, while evaluate,
device synchronization, and host-ready E/F/virial/stress outputs are inside the
timer. Correctness is checked only after all 500 measured outputs are buffered.
The reported score is the median of the three paired baseline/candidate
speedups; a value below 1 remains a valid PASS when protocol and correctness
pass. This is not an organizer-verified score and contains no private seeds,
runner, or validator.

`all` is exactly `build`, `test`, `benchmark`, then `package`. `image` only
writes a controlled `NOT_RUN` status and never contacts Bohrium. No command
creates, stops, deletes, or restarts a sandbox, pushes Git, builds an image, or
runs a private formal benchmark. Historical commits may be selected only with
an explicit immutable `--starter-ref`; the default is the frozen starter tag
declared above.

The runtime image recipe is a separate, later release step. It must clone this
public repository at the immutable starter tag during image construction; the
candidate source, model, structure, results and private evaluator are not image
acceptance evidence.
