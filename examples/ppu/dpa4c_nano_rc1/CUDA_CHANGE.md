# CUDA change

`candidate/dpa4c_contest_ops.cu` implements the DPA4C edge-gradient to force,
atom-virial and frame-virial scatter as a real CUDA/PPU device kernel. The
tracked adapter installs this operator only for inference; `create_graph=True`
continues to use the official DeepMD implementation.

The public smoke verifies E/F/virial/stress and reports a non-formal latency
preview. Final scoring is produced only by the organizer-controlled rebuild
and private paired protocol.
