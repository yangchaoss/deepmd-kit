FROM registry.dp.tech/dptech/dp/native/prod-87/training-xpu-pytorch:26.04.post1@sha256:20af455a90c11599de3dbec145083882ea8fe04ec37f4f57ce70a1dcbf5b0780

ENV PPU_SDK=/usr/local/PPU_SDK \
    CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
    CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK \
    CUDAToolkit_ROOT=/usr/local/PPU_SDK/CUDA_SDK \
    DPA4C_CONTESTANT_ROOT=/opt/dpa4c-contestant-kit

RUN set -eux; \
    test -x /opt/ac2/bin/python; \
    test -x "$CUDA_HOME/bin/nvcc"; \
    command -v git; \
    /opt/ac2/bin/python -c 'import torch; assert torch.__version__ == "2.9.0+ali.10.ppu2.1.0.cu130"; assert str(torch.version.cuda) == "13.0"'

RUN set -eux; \
    git -c http.proxy=http://ga.dp.tech:8118 -c http.version=HTTP/1.1 clone \
      --depth=1 --branch dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1 --single-branch \
      https://github.com/yangchaoss/deepmd-kit.git "$DPA4C_CONTESTANT_ROOT"; \
    git -C "$DPA4C_CONTESTANT_ROOT" checkout --detach HEAD; \
    test "$(git -C "$DPA4C_CONTESTANT_ROOT" rev-parse HEAD)" = 8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2; \
    test -z "$(git -C "$DPA4C_CONTESTANT_ROOT" status --porcelain)"; \
    ln -s "$DPA4C_CONTESTANT_ROOT/examples/ppu/dpa4c_nano_rc1/scripts/contestant.sh" /usr/local/bin/dpa4c-contestant-flow

LABEL org.opencontainers.image.source="https://github.com/yangchaoss/deepmd-kit" \
      com.dptech.dpa4c.candidate-commit="8b289e73cf0bfb1ff16f3ae3a30b88dc2cbb60e2" \
      com.dptech.dpa4c.candidate-tag="dpa4c-ppu-nano-contestant-kit-v1.0.0-rc1" \
      com.dptech.dpa4c.runtime-contract="ppu-sdk-2.1_cuda-13.0_torch-2.9.0"

WORKDIR /workspace
