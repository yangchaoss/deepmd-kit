FROM registry.dp.tech/dptech/dp/native/prod-27666/4553069/dpa4c-nano-rc1-runtime-probe:20260911-yangchao-r1@sha256:f475eb7a0488b8785818b506fc17e9c411b7ef526fdae2d974c62a5c42937061

ENV DPA4C_CONTESTANT_ROOT=/opt/dpa4c-contestant-kit

RUN set -eux; \
    test -x /opt/ac2/bin/python; \
    test -x "$CUDA_HOME/bin/nvcc"; \
    command -v git; \
    /opt/ac2/bin/python -c 'import torch; assert torch.__version__ == "2.9.0+ali.10.ppu2.1.0.cu130"; assert str(torch.version.cuda) == "13.0"'

RUN set -eux; \
    for attempt in 1 2 3; do \
      git -c http.version=HTTP/1.1 clone \
        --depth 1 \
        --branch dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1 \
        --single-branch \
        https://github.com/yangchaoss/deepmd-kit.git \
        "$DPA4C_CONTESTANT_ROOT" && break; \
      rm -rf "$DPA4C_CONTESTANT_ROOT"; \
      test "$attempt" -lt 3; \
      sleep 5; \
    done; \
    test "$(git -C "$DPA4C_CONTESTANT_ROOT" rev-parse HEAD)" = 8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2; \
    test "$(git -C "$DPA4C_CONTESTANT_ROOT" rev-parse HEAD^{tree})" = e4e5897d68bceeb7c05b0256bdfaa12bcf0002d2; \
    test -z "$(git -C "$DPA4C_CONTESTANT_ROOT" status --porcelain)"; \
    ln -s "$DPA4C_CONTESTANT_ROOT/examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh" /usr/local/bin/dpa4c-contestant-flow

LABEL org.opencontainers.image.source="https://github.com/yangchaoss/deepmd-kit" \
      com.dptech.dpa4c.candidate-commit="8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2" \
      com.dptech.dpa4c.candidate-tree="e4e5897d68bceeb7c05b0256bdfaa12bcf0002d2" \
      com.dptech.dpa4c.candidate-tag="dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1" \
      com.dptech.dpa4c.runtime-contract="ppu-sdk-2.1_cuda-13.0_torch-2.9.0"

WORKDIR /workspace
