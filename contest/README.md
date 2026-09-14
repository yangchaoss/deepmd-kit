# DPA4C PPU contestant source flow

The organizer calls one tracked entrypoint.  Assets remain external and are
accepted only at the frozen SHA-256 values in `config/runtime.json`.

```bash
./contest/contest.sh build --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
./contest/contest.sh test  --run-root /workspace/runs/OWNER/RUN_ID --assets-root /workspace/dpa4c-contest/assets
```

`build` creates a candidate virtual environment outside the checkout, builds a
non-editable wheel from the current committed tree, installs the wheel and the
small candidate session contract, then records baseline/candidate package and
ELF identities.  Before starting either worker, `test` re-records both runtime
identities and verifies stable package, ELF, Torch, and candidate-entry bindings
against `BUILD_STATUS.json`; mismatches fail closed in
`RUNTIME_IDENTITY_STATUS.json`.  Passing identities then run baseline and
candidate in separate processes from an external CWD and apply the frozen public
Nano/1024 FP32 E/F/virial/stress tolerances.  It does not run formal performance
or manage a sandbox or image.
