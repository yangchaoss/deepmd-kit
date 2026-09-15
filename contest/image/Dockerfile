FROM registry.dp.tech/dptech/dp/native/prod-27666/4553069/dpa4c-nano-rc1-runtime-probe:20260911-yangchao-r1@sha256:f475eb7a0488b8785818b506fc17e9c411b7ef526fdae2d974c62a5c42937061

ENV DPA4C_CONTESTANT_ROOT=/opt/dpa4c-contestant-kit \
    DPA4C_CONTEST_WHEELHOUSE=/opt/dpa4c-contest-wheelhouse \
    DPA4C_BASELINE_VENV=/opt/dpa4c-baseline-venv \
    DPA4C_BASELINE_BUILD=/opt/dpa4c-baseline-build \
    PPU_SDK=/usr/local/PPU_SDK \
    CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
    CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK \
    CUDAToolkit_ROOT=/usr/local/PPU_SDK/CUDA_SDK

RUN set -eux; \
    test -x /opt/ac2/bin/python; \
    test -x "$CUDA_HOME/bin/nvcc"; \
    command -v git; \
    /opt/ac2/bin/python -c 'import torch; assert torch.__version__ == "2.9.0+ali.10.ppu2.1.0.cu130"; assert str(torch.version.cuda) == "13.0"'

RUN set -eux; \
    rm -rf "$DPA4C_CONTESTANT_ROOT" "$DPA4C_CONTEST_WHEELHOUSE"; \
    for attempt in 1 2 3; do \
      git -c http.version=HTTP/1.1 clone \
        --depth 1 \
        --branch dpa4c-ppu-nano-starter-v1.0.0-rc6 \
        --single-branch \
        https://github.com/yangchaoss/deepmd-kit.git \
        "$DPA4C_CONTESTANT_ROOT" && break; \
      rm -rf "$DPA4C_CONTESTANT_ROOT"; \
      test "$attempt" -lt 3; \
      sleep 5; \
    done; \
    test "$(git -C "$DPA4C_CONTESTANT_ROOT" remote get-url origin)" = "https://github.com/yangchaoss/deepmd-kit.git"; \
    test "$(git -C "$DPA4C_CONTESTANT_ROOT" describe --exact-match --tags --match 'dpa4c-ppu-nano-starter-v1.0.0-rc6')" = "dpa4c-ppu-nano-starter-v1.0.0-rc6"; \
    if git -C "$DPA4C_CONTESTANT_ROOT" symbolic-ref -q HEAD >/dev/null; then \
      echo 'expected detached HEAD after immutable tag checkout' >&2; exit 1; \
    fi; \
    test -z "$(git -C "$DPA4C_CONTESTANT_ROOT" status --porcelain)"; \
    printf 'DPA4C_RESOLVED_HEAD=%s\n' "$(git -C "$DPA4C_CONTESTANT_ROOT" rev-parse HEAD)"; \
    printf 'DPA4C_RESOLVED_TREE=%s\n' "$(git -C "$DPA4C_CONTESTANT_ROOT" rev-parse HEAD^{tree})"

RUN set -eux; \
    mkdir -p "$DPA4C_CONTEST_WHEELHOUSE"; \
    /opt/ac2/bin/python -m pip download \
      --only-binary=:all: --no-deps --require-hashes \
      --dest "$DPA4C_CONTEST_WHEELHOUSE" \
      -r "$DPA4C_CONTESTANT_ROOT/contest/config/build-requirements.txt" \
      -r "$DPA4C_CONTESTANT_ROOT/contest/config/runtime-requirements.txt"; \
    /opt/ac2/bin/python -c 'import hashlib,json,pathlib; root=pathlib.Path("/opt/dpa4c-contestant-kit"); wheelhouse=pathlib.Path("/opt/dpa4c-contest-wheelhouse"); manifest=json.loads((root/"contest/config/wheelhouse-manifest.json").read_text()); items=manifest["locks"]["build"]["files"]+manifest["locks"]["runtime"]["files"]; bad=[item["filename"] for item in items if not (wheelhouse/item["filename"]).is_file() or hashlib.sha256((wheelhouse/item["filename"]).read_bytes()).hexdigest()!=item["sha256"]]; assert len(items)==24 and not bad, (len(items),bad)'

RUN set -eux; \
    /opt/ac2/bin/python -m pip install \
      --no-index --find-links "$DPA4C_CONTEST_WHEELHOUSE" \
      --no-deps --require-hashes --ignore-installed \
      -r "$DPA4C_CONTESTANT_ROOT/contest/config/runtime-requirements.txt"; \
    ln -s "$DPA4C_CONTESTANT_ROOT/contest/contest.sh" /usr/local/bin/dpa4c-contestant-flow; \
    test -x /usr/local/bin/dpa4c-contestant-flow

# Build and install the frozen baseline outside the tracked starter checkout.
# The candidate flow later builds its own wheel into its own run root; no
# candidate artifact is installed into this baseline venv.
RUN set -eux; \
    rm -rf "$DPA4C_BASELINE_VENV" "$DPA4C_BASELINE_BUILD"; \
    /opt/ac2/bin/python -m venv --system-site-packages "$DPA4C_BASELINE_VENV"; \
    baseline_python="$DPA4C_BASELINE_VENV/bin/python"; \
    "$baseline_python" -m pip install \
      --no-index --find-links "$DPA4C_CONTEST_WHEELHOUSE" \
      --no-deps --require-hashes --ignore-installed \
      -r "$DPA4C_CONTESTANT_ROOT/contest/config/build-requirements.txt"; \
    mkdir -p "$DPA4C_BASELINE_BUILD/wheels"; \
    SKBUILD_BUILD_DIR="$DPA4C_BASELINE_BUILD/scikit-build" \
      DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 \
      "$baseline_python" -s -m pip wheel "$DPA4C_CONTESTANT_ROOT" \
        --no-deps --no-build-isolation --wheel-dir "$DPA4C_BASELINE_BUILD/wheels"; \
    set -- "$DPA4C_BASELINE_BUILD"/wheels/deepmd_kit-*.whl; \
    test "$#" -eq 1; \
    baseline_wheel="$1"; \
    "$baseline_python" -m pip install --no-index --no-deps --force-reinstall "$baseline_wheel"; \
    "$baseline_python" -m pip install \
      --no-index --find-links "$DPA4C_CONTEST_WHEELHOUSE" \
      --no-deps --require-hashes --ignore-installed \
      -r "$DPA4C_CONTESTANT_ROOT/contest/config/runtime-requirements.txt"; \
    test -z "$(git -C "$DPA4C_CONTESTANT_ROOT" status --porcelain)"; \
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$baseline_python" -c 'import hashlib,json,subprocess,sys; from pathlib import Path; import deepmd; import deepmd.lib; from deepmd.calculator import DP; prefix=Path(sys.prefix).resolve(); source=Path("/opt/dpa4c-contestant-kit").resolve(); wheel=Path("'"$baseline_wheel"'").resolve(); d=Path(deepmd.__file__).resolve(); l=Path(deepmd.lib.__file__).resolve(); assert d.is_relative_to(prefix) and l.is_relative_to(prefix); assert not d.is_relative_to(source) and not l.is_relative_to(source); elfs=[{"path":str(p.resolve()),"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest(),"size_bytes":p.stat().st_size} for p in sorted(l.parent.rglob("*.so"))]; assert elfs,"baseline deepmd.lib contains no shared objects"; result={"status":"PASS","python":str(Path(sys.executable).resolve()),"prefix":str(prefix),"deepmd":str(d),"deepmd_lib":str(l),"deepmd_elfs":elfs,"baseline_wheel":{"path":str(wheel),"sha256":hashlib.file_digest(wheel.open("rb"),"sha256").hexdigest(),"size_bytes":wheel.stat().st_size},"starter":{"repository":"https://github.com/yangchaoss/deepmd-kit.git","tag":"dpa4c-ppu-nano-starter-v1.0.0-rc6","commit":subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip(),"tree":subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD^{tree}"],text=True).strip()},"sys_path":sys.path}; Path("/opt/dpa4c-baseline-identity.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))'

LABEL org.opencontainers.image.source="https://github.com/yangchaoss/deepmd-kit" \
      org.opencontainers.image.version="dpa4c-ppu-nano-starter-v1.0.0-rc6" \
      com.dptech.dpa4c.source-tag="dpa4c-ppu-nano-starter-v1.0.0-rc6" \
      com.dptech.dpa4c.source-provenance="resolved-head-tree-in-build-log-and-release-manifest" \
      com.dptech.dpa4c.runtime-base="registry.dp.tech/dptech/dp/native/prod-27666/4553069/dpa4c-nano-rc1-runtime-probe:20260911-yangchao-r1@sha256:f475eb7a0488b8785818b506fc17e9c411b7ef526fdae2d974c62a5c42937061" \
      com.dptech.dpa4c.runtime-contract="ppu-sdk-2.1_cuda-13.0_torch-2.9.0" \
      com.dptech.dpa4c.wheelhouse="24-wheel-hash-locked"

WORKDIR /workspace
