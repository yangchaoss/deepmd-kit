# DPA4C PPU contestant source flow (authoritative)

This file is the only authoritative contestant-facing contract for the Nano
PPU flow. The starter repository is
`https://github.com/yangchaoss/deepmd-kit.git`; the immutable starter ref is
`dpa4c-ppu-nano-starter-v1.0.0-rc5`. Resolve that tag to record the exact
commit and tree used by a run. Historical files under
`examples/ppu/dpa4c_nano_rc1/` are compatibility material only and do not
define a second submission contract.

The published `dpa4c-ppu-nano-starter-v1.0.0-rc1`,
`dpa4c-ppu-nano-starter-v1.0.0-rc2`, and
`dpa4c-ppu-nano-starter-v1.0.0-rc3` tags are superseded historical material
and must not be used as the current starter. rc2 was superseded after final
Runtime Image prevalidation exposed candidate dependency inheritance and
failed the isolation gate; rc1, rc2, and rc3 remain history only.

The organizer calls one tracked entrypoint.  Assets remain external and are
accepted only at the frozen SHA-256 values in `config/runtime.json`.

## Short contestant flow

```text
clone the public starter (or enter the repository already present in the runtime image)
  -> mount the two fixed assets under --assets-root
  -> ./contest/contest.sh all --profile quick
  -> make and commit the candidate change
  -> ./contest/contest.sh all --profile full
  -> submit the generated eight-file submission directory
```

The organization provides or mounts these files; no download URL is implied:

| file | expected SHA-256 |
|---|---|
| `DPA4C-Nano-OMat24-v20260819.pt` | `f894ac16adfb7f5030d4fe4e2849db6c608f9c7074badfafb4a50c9b9afed00a` |
| `common-structure-1024.extxyz` | `137056e51cf63bd7dabf0508a29959218baf109da0c5e19a3765d11873c891e7` |

Example asset root: `/workspace/dpa4c-contest/assets` (or pass another
`--assets-root`). Missing files or SHA mismatches fail before build/compute.

The measured-output contract is controlled by `config/output-contract.json`
and is recorded by the result, repeats, and measurement binding. It requires
the exact six archive fields: four physics fields (`energy`, `forces`, `virial`,
`stress`) plus `warmup_latencies_s` and `measured_latencies_s`. Physics uses
measured dimension `N`, timing uses warmup dimension `W` and measured `N`, and
all arrays are host `float64`; timing values must be finite and strictly
positive. The physics shapes are
`(N,)`, `(N,1024,3)`, `(N,3,3)`, and `(N,6)`. Contract checks run before
numeric tolerance checks; reference/baseline contract failures invalidate the
benchmark and candidate contract failures invalidate the candidate.

```bash
./contest/contest.sh build --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh test  --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh benchmark --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets --starter-ref dpa4c-ppu-nano-starter-v1.0.0-rc5
./contest/contest.sh package --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets --starter-ref dpa4c-ppu-nano-starter-v1.0.0-rc5
```

The default profile is `quick`: one public baseline/candidate pair with 2
warmup and 10 measured frames. Its result is marked
`score_type=development_quick`, `verified=false`, and
`formal_performance=NOT_RUN`; it never creates a submission. `full` retains
the public self-test protocol of two fresh pairs in AB/BA order with 20
warmup and 100 measured frames. Both profiles use the same worker, timer,
synchronization, input generation, and E/F/virial/stress checks.

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

`benchmark` is a public, unverified self-test. The `full` profile runs two
fresh-process baseline/candidate pairs in AB/BA order, with 20 warmup and 100
measured frames per route. The two pairs use disjoint public input sequences;
within each pair, baseline/candidate/reference share the same inputs. Each
route uses a separate process; reference evaluation is also separate. Host
inputs are materialized before timing, while evaluate, device synchronization,
and host-ready E/F/virial/stress outputs are inside the timer. Correctness is
checked after all measured outputs are buffered. The two paired speedups are
retained, and full's headline is the geometric mean
`sqrt(S_AB*S_BA)` (`paired_geometric_mean_speedup`), never the best pair. Every
route also records mean, p50, p90, p99, CV, and throughput; p99 is diagnostic
only. This is not an organizer-verified score and contains no private seeds,
runner, or validator.

The `all` command expands in exactly this order: `build` → `test` →
`benchmark` → `package` (the package stage is explicitly skipped for quick,
or for full with no candidate change). `image` only
writes a controlled `NOT_RUN` status and never contacts Bohrium. No command
creates, stops, deletes, or restarts a sandbox, pushes Git, builds an image, or
runs a private formal benchmark. Historical commits may be selected only with
an explicit immutable `--starter-ref`; the default is the frozen starter tag
declared above.

The public aggregate `result.json` is a compact summary: it contains the
explicit protocol (`pair_count`, `pair_order`, `warmup`, `measured`, and
`aggregation_method`), pair latency/throughput aggregates, correctness maxima
and tolerances, and the two pair speedups plus the selected headline. The
complete per-frame correctness and route evidence remains in `repeats.json`,
which repeats the same protocol metadata; packaging binds the compact result
SHA rather than duplicating those arrays. The fixed submission manifest also
records the protocol and aggregation method. A full run with no committed
candidate change may complete measurement but is explicitly
`PACKAGE_SKIPPED`; only a changed, passing full run creates the eight-file
submission.

The runtime image recipe is a separate, later release step. It must clone this
public repository at the immutable starter tag during image construction; the
candidate source, model, structure, results and private evaluator are not image
acceptance evidence.
