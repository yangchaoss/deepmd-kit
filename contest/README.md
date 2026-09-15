# DPA4C Nano PPU 参赛教程

本目录是 DPA4C Nano 单 PPU 推理优化赛的选手入口。目标是在不改变模型、输入、FP32 精度和 E/F/virial/stress 数学语义的前提下，修改 DPA4C GPU 执行路径并提高端到端推理性能。

当前唯一有效版本：

```text
repository:       https://github.com/yangchaoss/deepmd-kit.git
starter ref:      dpa4c-ppu-nano-starter-v1.0.0-rc7
source checkout:  /opt/dpa4c-contestant-kit
assets root:      /workspace/dpa4c-contest/assets
run roots:        /workspace/runs/<owner>/...
wheelhouse:       /opt/dpa4c-contest-wheelhouse
```

历史 rc1–rc6 仅保留作归档，不是本轮提交基线。

## 1. 你需要完成什么

选手可以修改完整 DPA4C GPU execution path，包括 neighbor、descriptor、network、force、virial、聚合、内存布局、调度和 kernel fusion。最终需要证明：

1. 源码能从冻结 starter 和 `candidate.patch` 还原；
2. 源码能在统一 PPU Runtime Image 中重新构建；
3. 运行时实际加载本次构建的 candidate；
4. 候选包含真实 CUDA C/C++ device 实现并在 PPU 上执行；
5. 每个 measured frame 的 E/F/virial/stress 均通过校验；
6. 公开性能结果与本次源码、二进制、输入和协议绑定。

外部调用保持 eager，但这只是评测接口约束，不要求候选内部保留原 PyTorch operator graph。未修改部分可以继续使用冻结的 PyTorch 或平台基础库。

## 2. 镜像、资产与目录

### 2.1 Runtime Image 提供什么

组织方 Runtime Image 提供：

- PPU SDK、CUDA 兼容工具链、PyTorch 和基础系统依赖；
- 冻结 baseline 环境 `/opt/dpa4c-baseline-venv`；
- starter 源码 `/opt/dpa4c-contestant-kit`；
- 离线依赖包 `/opt/dpa4c-contest-wheelhouse`；
- 一键入口 `/usr/local/bin/dpa4c-contestant-flow`。

Runtime Image 是统一运行环境，不是 candidate 实现。选手仍需修改并提交 Git 源码。

### 2.2 外部资产

组织方将以下只读资产挂载到 `/workspace/dpa4c-contest/assets`：

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `DPA4C-Nano-OMat24-v20260819.pt` | 固定模型 | `f894ac16adfb7f5030d4fe4e2849db6c608f9c7074badfafb4a50c9b9afed00a` |
| `common-structure-1024.extxyz` | 固定 1024 原子周期结构 | `137056e51cf63bd7dabf0508a29959218baf109da0c5e19a3765d11873c891e7` |

不要下载、修改或提交这些资产。文件缺失或 SHA 不符时，流程会在构建或计算前停止。

### 2.3 目录地图

```text
/opt/dpa4c-contestant-kit/          Git源码，只放受控实现
├── deepmd/                          Python实现
├── source/                          C++/CUDA与构建源码
├── contest/
│   ├── candidate/                   候选加载和适配入口，可修改
│   ├── config/                      冻结协议，不可修改
│   ├── scripts/                     公开评测器，不可修改
│   ├── tests/                       工具链测试，不可修改
│   ├── image/                       Runtime Image配方，不可修改
│   ├── contest.sh                   统一入口，不可修改
│   └── README.md                    本教程，不可修改
└── README.md                        上游项目入口，不可修改

/workspace/dpa4c-contest/assets/     模型与结构，只读
/workspace/runs/<owner>/...          构建、日志、结果和提交包
/opt/dpa4c-contest-wheelhouse/       冻结离线依赖
```

运行结果和虚拟环境放在 `/workspace/runs`，不要写进 Git checkout。

## 3. 五分钟跑通基线

进入 Runtime Image 后：

```bash
cd /opt/dpa4c-contestant-kit
git switch --detach dpa4c-ppu-nano-starter-v1.0.0-rc7
git switch -c candidate/my-model
git config user.name "Contestant"
git config user.email "contestant@example.invalid"

/usr/local/bin/dpa4c-contestant-flow all --profile quick
```

Quick 自动创建唯一运行目录，并完成：

```text
build -> test -> benchmark
```

Quick 使用 **1 pair、AB order、2 warmup + 10 measured**。它用于验证构建、运行身份、E/F/virial/stress 和基本性能，不生成提交包。

命令结束时会打印 `run_root`。重点查看：

```text
<run_root>/FLOW_STATUS.json
<run_root>/results/public-benchmark/result.json
```

结果含义：

- `status=PASS`：本次公开流程通过；
- `paired_speedup > 1`：候选比冻结 baseline 快；
- `paired_speedup < 1`：候选比冻结 baseline 慢；
- `verified=false`：这是公开自测，不是正式成绩。

## 4. 修改候选实现

最小候选入口是 [`candidate/session.py`](candidate/session.py)：

```python
def create_session(*, model: str):
    ...

class CandidateSession:
    def evaluate(self, atoms) -> dict[str, object]:
        ...
```

`create_session()` 只负责创建和加载候选实现。输入提供、调用顺序、设备同步、计时、正确性校验和结果保存均由组织方 runner 控制。

允许修改：

- `deepmd/**`、`source/**`、CMake 和项目安装配置；
- 新增或修改 `.cu/.cuh/.cpp/.h/.py` 源文件；
- `contest/candidate/**` 中的候选加载和适配代码。

禁止 candidate 修改：

```text
README.md
contest/README.md
contest/contest.sh
contest/config/**
contest/scripts/**
contest/tests/**
contest/image/**
```

除 `contest/candidate/**` 外，全部 `contest/**` 均为保护路径。工具会检查新增、删除、重命名和权限变化。

正式实现必须由源码重新构建，不能用 `.so`、`.o`、`.a` 或 `.whl` 作为实现来源。不要修改计时器、输入、reference、baseline、validator、结果聚合或输出逻辑。

## 5. 日常开发流程

每轮修改后先提交，再跑 Quick：

```bash
git status --short
git add <changed-files>
git commit -m "optimize DPA4C Nano on PPU"

/usr/local/bin/dpa4c-contestant-flow all --profile quick
```

每次 `all` 都会新建 owner 隔离目录，不覆盖旧结果。源码没有 commit 时，不能形成可提交的 patch 和运行身份。

Quick 稳定 PASS 后再运行 Full：

```bash
/usr/local/bin/dpa4c-contestant-flow all --profile full
```

Full 使用 **2 fresh pairs、AB/BA order、20 warmup + 100 measured**。baseline、candidate 和 reference 使用独立进程；两个 pair 使用不同输入序列，同一 pair 内三条路线使用完全相同的输入。

## 6. 正确性门禁

固定测试条件：

| 项目 | 值 |
|---|---|
| 模型 | DPA4C Nano OMat24 |
| 结构 | 1024 原子周期结构 |
| 精度 | FP32 |
| 输出 | energy、forces、virial、stress |

每个 measured frame 的实际输出都会缓存在 timer 外统一与 reference 比较。比较前先检查字段、shape、dtype 和有限值，再使用 [`config/benchmark-tolerance.json`](config/benchmark-tolerance.json) 中冻结的公开容差：

| 字段 | atol | rtol |
|---|---:|---:|
| energy (eV) | `3.9577481061314757e-4` | `1.4210854715202004e-14` |
| forces (eV/A) | `5.0e-5` | `1.4210854715202004e-14` |
| virial (eV) | `6.802242146053405e-4` | `1.4210854715202004e-14` |
| stress (eV/A^3) | `1.0e-7` | `1.4210854715202004e-14` |

reference 或 baseline 失败时，本轮为 `BENCHMARK_INVALID`；candidate 失败时为 `CANDIDATE_INVALID`。两者均不产生有效候选性能结果。

公开输出合同由 [`config/output-contract.json`](config/output-contract.json) 固定。物理量 shape 分别为 `(N,)`、`(N,1024,3)`、`(N,3,3)` 和 `(N,6)`；归档数组为 host `float64`。这是结果归档格式，不表示模型从 FP32 改成 FP64。

## 7. 性能与公开自测分数

计时范围：

```text
组织方准备当前host positions/cell
================ TIMER START ================
candidate.evaluate(atoms)
  H2D
  neighbor/update
  model与自定义CUDA执行
  device synchronization
  D2H
  host E/F/virial/stress ready
================ TIMER STOP =================
```

模型加载、首次构建、warmup、reference 计算、校验和日志不计入 measured 时间。

单个 pair 的加速比：

```text
S_pair
= candidate throughput / baseline throughput
= baseline elapsed time / candidate elapsed time
```

Full 的公开自测主结果：

```text
public self-test speedup = sqrt(S_AB * S_BA)
```

`1.20x` 表示本次公开自测约快 20%，`1.00x` 表示基本持平。mean、p50、p90、p99、CV 和 throughput 同时保留；p99 与 CV 用于诊断长尾和波动，不单独作为主分数。

公开结果始终标记：

```text
score_type=public_self_test
verified=false
formal_performance=NOT_RUN_BY_SCOPE
```

正式评测由组织方在同一冻结 Runtime Image 中从 `candidate.patch` 重新构建，并使用 **3 组独立 fresh-process pairs、AB/BA/AB order、每条路线 20 warmup + 500 measured**。三组 pair 使用互不重叠的私有 measured 序列，同一 pair 内 baseline、candidate 和 reference 使用相同输入。

每组正式加速比仍按 `S_i = baseline elapsed / candidate elapsed` 计算，正式性能主值为：

```text
formal speedup = median(S_1, S_2, S_3)
```

三个 pair 的正确性必须全部 PASS；任一 candidate measured frame 失败，该 candidate 不产生正式成绩。选手的公开结果只是预览，不直接成为排名成绩；私有输入、reference 输出、正式 runner 和 validator 不随 starter 发布。若比赛平台后续将 speedup 映射为积分或名次，以组织方发布的赛事计分规则为准，本仓库不自行推算平台积分。

## 8. 一键生成提交包

Full 只有同时满足以下条件才生成提交目录：

1. candidate 相对 rc7 有已提交的源码变化；
2. 未触碰保护路径，未提交禁止的二进制实现；
3. runtime identity、正确性和 Full benchmark 全部 PASS；
4. 当前 committed tree 与实际 measured tree 完全一致。

提交目录固定只有八个文件：

```text
candidate.patch
result.json
repeats.json
measurement-binding.json
submission-manifest.json
image.json
CHANGELOG.md
SHA256SUMS
```

| 文件 | 内容 |
|---|---|
| `candidate.patch` | 从 rc7 到 measured commit 的源码变化 |
| `result.json` | 公开自测精简结果 |
| `repeats.json` | pair、进程与逐帧正确性证据 |
| `measurement-binding.json` | 源码、构建、运行产物、输入和协议身份 |
| `submission-manifest.json` | 提交合同和文件清单 |
| `image.json` | Runtime Image 身份 |
| `CHANGELOG.md` | 选手的改动说明 |
| `SHA256SUMS` | 文件完整性校验 |

提交前检查：

```bash
git status --short
cd <generated-submission-directory>
sha256sum -c SHA256SUMS
find . -maxdepth 1 -type f -print | sort
```

提交完整生成目录，不要手工修改 JSON、patch 或 SHA256SUMS。源码再次变化后必须重新 commit、重新 Full，不能用新源码包装旧结果。

## 9. 分阶段调试

正常开发优先用 `all`。只有定位问题时才使用分阶段命令，并始终显式指定同一个目录：

```bash
RUN_ROOT=/workspace/runs/OWNER/RUN_ID
ASSETS_ROOT=/workspace/dpa4c-contest/assets

./contest/contest.sh build \
  --run-root "$RUN_ROOT" --assets-root "$ASSETS_ROOT"
./contest/contest.sh test \
  --run-root "$RUN_ROOT" --assets-root "$ASSETS_ROOT"
./contest/contest.sh benchmark \
  --run-root "$RUN_ROOT" --assets-root "$ASSETS_ROOT" \
  --profile full --starter-ref dpa4c-ppu-nano-starter-v1.0.0-rc7
./contest/contest.sh package \
  --run-root "$RUN_ROOT" --assets-root "$ASSETS_ROOT" \
  --profile full --starter-ref dpa4c-ppu-nano-starter-v1.0.0-rc7
```

| 阶段 | 功能 |
|---|---|
| `build` | 从 committed tree 构建非 editable wheel，安装到独立 candidate prefix |
| `test` | 检查实际 import、loaded ELF、Torch 和 candidate entrypoint，再做基础正确性测试 |
| `benchmark` | 执行公开配对性能测试并绑定实际输出 |
| `package` | 检查 measured tree 一致性并生成八文件提交目录 |

依赖严格从 `/opt/dpa4c-contest-wheelhouse` 离线、按 hash 安装。选手不需要联网安装依赖。只有组织方调试时才可通过 `DPA4C_CONTEST_WHEELHOUSE` 使用其他受控 wheelhouse。

## 10. 功能边界与已知限制

- 当前 starter 只覆盖 **DPA4C Nano、1024 原子、FP32、单 PPU 推理**；Mini/Neo、训练和 LAMMPS 不在本轮范围。
- Quick 是低成本开发检查，不能代表稳定性能；Full 是公开自测，也不是正式排名。
- 正式 CUDA authenticity、私有输入复测和最终排名由组织方完成，公开工具不包含私有裁判逻辑。
- 公开 tolerance 是冻结 baseline 在当前 Nano/1024/FP32 协议下的重复性边界，不是科学精度声明，也不能迁移到其他模型或精度。
- `all` 不会创建、停止、删除或重启 Sandbox，不会 push Git，也不会构建 Bohrium Image。
- `image` 子命令只记录 `NOT_RUN`；Runtime Image 由组织方独立维护。
- candidate 不控制正式 benchmark 循环、输入生成、同步、reference、validator、计时器、聚合和结果写入。
- candidate 源码和结果不会被写入 Runtime Image；镜像负责环境，Git patch 负责实现。

遇到失败时，先查看 `<run_root>/FLOW_STATUS.json` 和对应阶段日志。不要在失败后手工补写结果文件；修复源码后新建 commit 并重新运行。
