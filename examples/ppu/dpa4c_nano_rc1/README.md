# Legacy DPA4C Nano compatibility path

The authoritative contestant contract is
[`contest/README.md`](../../../contest/README.md). Use its single tracked
entrypoint, `contest/contest.sh`, and the immutable starter repository/ref
declared there:

```text
https://github.com/yangchaoss/deepmd-kit.git
dpa4c-ppu-nano-starter-v1.0.0-rc2
```

The published `dpa4c-ppu-nano-starter-v1.0.0-rc1` tag is superseded
historical material and must not be used as the current starter.

`scripts/contestant.sh` is retained only as a thin compatibility forwarder to
`contest/contest.sh`. It does not define an alternate build, test, benchmark,
score, package, image or submission contract. The older helper scripts,
historical source lock and image recipes in this directory are preserved for
provenance and are not the current starter or formal acceptance path; do not
use their old rc1 checkout instructions.

The public flow keeps model, structure, private evaluator data, credentials,
results and build products outside Git. Formal seeds, reference outputs and
official scoring remain organizer-private.
