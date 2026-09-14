FROM registry.dp.tech/dptech/dp/native/prod-87/training-xpu-pytorch:26.04.post1@sha256:20af455a90c11599de3dbec145083882ea8fe04ec37f4f57ce70a1dcbf5b0780

ENV PPU_SDK=/usr/local/PPU_SDK \
    CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
    CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK \
    CUDAToolkit_ROOT=/usr/local/PPU_SDK/CUDA_SDK

RUN set -eux; \
    test -x /opt/ac2/bin/python; \
    test -x "$CUDA_HOME/bin/nvcc"; \
    /opt/ac2/bin/python -c 'import torch; assert torch.__version__ == "2.9.0+ali.10.ppu2.1.0.cu130"; assert str(torch.version.cuda) == "13.0"'

LABEL org.opencontainers.image.source="https://github.com/yangchaoss/deepmd-kit" \
      com.dptech.dpa4c.runtime-contract="ppu-sdk-2.1_cuda-13.0_torch-2.9.0"

WORKDIR /workspace
