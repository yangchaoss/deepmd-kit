# DPA4C PPU contestant source flow

The organizer calls one tracked entrypoint.  Assets remain external and are
accepted only at the frozen SHA-256 values in `config/runtime.json`.

```bash
./contest/contest.sh build --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh test  --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh benchmark --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets --starter-ref STARTER_TAG_OR_COMMIT
./contest/contest.sh package --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets --starter-ref STARTER_TAG_OR_COMMIT
```

`build` creates a candidate virtual environment outside the checkout, builds a
non-editable wheel from the current committed tree, installs the wheel and the
small candidate session contract, then records baseline/candidate package and
ELF identities.  Before starting either worker, `test` re-records both runtime
identities and verifies stable package, ELF, Torch, and candidate-entry bindings
against `BUILD_STATUS.json`; mismatches fail closed in
`RUNTIME_IDENTITY_STATUS.json`.  Passing identities then run baseline and
candidate in separate processes from an external CWD and apply the frozen public
Nano/1024 FP32 E/F/virial/stress tolerances.

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
runs a private formal benchmark. Until the release tag exists, pass an explicit
immutable `--starter-ref`; the default future release tag is
`dpa4c-ppu-nano-starter-v1.0.0-rc1`.
