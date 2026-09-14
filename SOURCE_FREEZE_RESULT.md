# DPA4C Nano PPU source freeze

This document is part of the frozen starter source. Resolve the immutable tag
below to obtain the final commit and tree; the values are intentionally derived
from Git so this file cannot become a second source of identity.

```text
repository: https://github.com/yangchaoss/deepmd-kit.git
starter_tag: dpa4c-ppu-nano-starter-v1.0.0-rc1
commit: git rev-parse dpa4c-ppu-nano-starter-v1.0.0-rc1^{commit}
tree: git rev-parse dpa4c-ppu-nano-starter-v1.0.0-rc1^{tree}
frozen_base: 3d079bdfb3d5ba8b5604bdedff7d196082f7d80b
```

`contest/README.md` is the single authoritative contestant-facing contract.
The `contest/contest.sh` entrypoint remains the only build, test, benchmark,
package and image-status flow. The example README and contestant script are
compatibility forwarding material only.

Verified before publication:

- tracked source is clean and contains no untracked implementation or binary
  build artifact;
- E3 wheelhouse, runtime lock, public smoke tolerance and one-click flow are
  unchanged;
- local static checks and the existing fast contract tests pass;
- candidate Golden source, private evaluator, formal seeds, reference outputs,
  formal 3x20+500 performance, runtime image and final score are not included
  or claimed here.

This freeze is source-only. A later Runtime Image stage may clone the public
repository at the immutable starter tag; it must not treat candidate data,
models, results or historical build products as image acceptance evidence.
