# ROCm GDN / PLE 内部 prefill checkpoint

vLLM 侧的改动只位于 `vllm-plugin-das`，通过 worker 的精确模块回调适配部署版
vLLM 0.28.1。它让一个 prefill forward 在内部保存可复用的 recurrent state
和卷积 history，随后继续计算剩余 token。

## 启用

在启动 prefill worker 前设置：

```bash
export VLLM_HCU_MAMBA_PREFILL_CHECKPOINT=1
```

同时选择 `--gdn-prefill-backend triton`。插件 CLI 最终会把该选项传入
`additional_config.gdn_prefill_backend`；直接使用 Python API 时可设置
`additional_config={"gdn_prefill_backend": "triton"}`。需要启用 prefix caching，
并使用 `mamba_cache_mode="align"`。

PD 部署的 P / D 都应设置同一开关，并选择同一 prefill backend，使两端的
cache spec 和 block table 规划保持一致。

开关默认关闭。只有所有 Mamba 层都声明支持时，GDN / PLE 的实际 cache spec
才会声明一个额外 checkpoint block，alignment 为 64。未知 Mamba 层或超过
64 token 的卷积 history 会让本次模型配置继续使用原有切分逻辑。没有加载项目
checkpoint 扩展的 AITER 不声明这项能力；decode backend 不受该选项影响。

2026-11-19 BoltOps checkpoint 实现已拆到独立目录：
`/jsyadmin/guanyu/qwen3_8-next-flash-PD/boltops-gdn-kernel`。将该目录加入
`PYTHONPATH`，选择 `--gdn-prefill-backend hcu-boltops` 并启用上述开关，插件
会导入外置 `boltops_gdn_kernel`，使用同一 Triton 实现和 gfx936 H24/H48 配置。
无需重编译或重新安装 HCU，首次运行只由 Triton 自动 JIT 编译并缓存。
插件中仅保留导入入口，kernel 和配置不进入 HCU 构建或分发。
FP32 cache 由 recurrence 直接写 slot，BF16 cache 经 FP32 snapshot 转换。
未加载外置包时不声明 BoltOps checkpoint 能力，继续原切分逻辑。
直接调用示例见外置目录 README.md。普通/prof/tp2dp1ep2 的 P/D 启动脚本
已加入该搜索路径；实际生效需要启动新进程。
原优化记录见
`/jsyadmin/guanyu/qwen3_8-next-flash-PD/boltops-gdn-chunk-delta-h-optimizer/gdn_checkpoint/README.md`。

项目 AITER HIP 扩展位于 `/jsyadmin/guanyu/qwen3_8-next-flash-PD/aiter-kernel`。
将该目录加入 P/D 两端的 `PYTHONPATH`，选择 `--gdn-prefill-backend hcu-aiter`
并启用上述开关，即可使用原生 HIP/CK Tile recurrence 的 FP32 checkpoint。
当前支持 gfx936、K=V=128 和转置 state layout。扩展不覆盖系统 AITER 安装包。

## 执行流程

1. 每个 cache group 的 metadata builder 使用自己的 block table 解析目标 slot，
   沿用部署版 `get_mamba_prefill_checkpoint_position` 与 validity helper。
   speculative-decode 请求不参与 prefill checkpoint 规划。
2. GDN 在卷积调用前，从原始 QKV 输入保存目标位置的完整 conv history。
   history 长度取层的 `conv_kernel_size - 1`，兼容插件转置后的 NN-layout 权重；
   在 GPU launch 前校验输入与 state 的 channel / history 维度。
3. checkpoint 专用 Triton recurrence 从 FP32 累加器写出目标位置的 state，
   然后继续同一次 recurrence；不从 BF16 的中间 `h` 恢复 FP32 state。
4. recurrent state 写入预留 slot；普通的 final-state 写回仍由 vLLM 执行。
   conv / recurrent slot 地址均使用 64 位乘法，支持大缓存池和非连续的大 stride，
   避免 BF16 缓存超过 4 GiB 时发生 32 位元素偏移溢出。
5. AMD PLE 同样保存对应位置的 history。原版和插件 fused PLE 都通过各自的
   `_short_conv_dilated_dispatch` 接入，支持非连续 state layout 和混合 batch 的
   token 索引映射。
6. 部署版缓存管理器负责按 checkpoint 的实际前缀位置注册 hash，供后续命中。

## 范围

复用现有调度器和缓存管理器。调度器仍要求本轮覆盖剩余 prefill 才豁免相关切分；
例如从 220736 处理到 256000，预算足够时可以一次处理 35264 token，并内部保存
checkpoint。若本轮只允许 28800 token，原有逻辑仍可能先处理到下一块边界，
产生一个 64-token 小步。

本改动不扩展非末轮 prefill 的调度策略，不提高 token budget，也不改变 PD
connector 的传输协议。首次启用需要编译 Triton kernel；实际收益取决于 token
budget、命中位置、并发和 backend，应在部署配置下测量。

## 验证

```bash
PYTHONPATH=../vllm:$PWD HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=__disabled__ python -m pytest -q \
  tests/runtime_patch/test_mamba_prefill_checkpoint.py
```

测试覆盖内部 state 与截断 forward 的结果、恢复后的继续计算、FP32 / BF16
cache、卷积 history、layout / token 映射、混合 batch metadata、各 group 的独立
slot、全模型能力门控、异常时 context 清理，以及现有调度器的预算限制。

`tools/analyze_mamba_checkpoint_trace.py` 可检查实际 torch trace：

```bash
python tools/analyze_mamba_checkpoint_trace.py TRACE.pt.trace.json.gz \
  --output checkpoint-trace.json --require-checkpoint
```

脚本用 runtime correlation ID 将 GPU kernel 归属到 `execute_context`，输出每个
forward 的 token 数和 conv / recurrent checkpoint 导出次数。必须同时看到两类
导出，才能证明 GDN 的两种 state 都走了新路径；仅出现 FLA recurrence 名称不足以
证明内部 checkpoint 已启用。

上述 `--require-checkpoint` 判断适用于 recurrent state 有独立 copy kernel 的路径。
外置 BoltOps 直接写 FP32 cache 时会省略 `_store_recurrent_checkpoint`，当前分析器
不会把普通 recurrence 名称当作导出证据。因此该路径不能依赖这个选项验收；需要
结合已启用的 checkpoint 参数、目标 cache 内容和续算结果验证。本次独立 kernel
profile 与相应 correctness 测试位于上文的 `boltops-gdn-chunk-delta-h-optimizer` 目录。
项目 AITER HIP 扩展的 kernel 有专用名称 `aiter_gdn_checkpoint_kernel`；分析器
已将它识别为融合的 recurrent export，可与 conv export 一起接受
`--require-checkpoint` 验证，普通 AITER recurrence 名称不会触发这一判断。

## 上游来源

`fla_prefill_checkpoint.py` 的 recurrence 基于本地上游 vLLM checkout
`b5b9f7982c` 中的 `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py`。
GDN 层接入设计参考 `29c6cfc457`（2026-11-17，PR #65186）。本地保留部署版
cache spec API，没有迁移上游新版 manager-config API。

## 本机验证（2026-11-18）

- gfx936，P1 / P2 各 DP2、D 为 DP4，沿用 `2p1d-agent-loop/prof` 的脚本和模型。
- P 的 token budget 为 38400、GPU memory utilization 为 0.93；D 为 256 / 0.90。
- 24 项新增测试通过，覆盖真实 GDN / PLE dispatch、NN-layout 转置权重、
  FP32 / BF16 checkpoint、GQA 混合长度请求，以及跨 32 位元素偏移的大 stride。
- 25 项既有 GDN / AITER 相关回归通过。
- 按 `prof/prof_pd_once.sh 5 256000 20 85` 抓取了四个 P rank、四个 D rank
  的 trace；正式轮 5 个请求全部成功，实际 prefix hit 为 86.225337%。
- 四个 P rank 的每个非空 prefill forward 都是 35263 token，总计 5 次；
  每次 forward 导出 37 次 conv checkpoint（36 个 GDN + 1 个 PLE），
  以及 36 次 recurrent checkpoint。原 trace 中的
  `64 + 28800 + 6272 + 127` 四次 forward，在本次配置下成为一次。
- 本次正式轮 mean TTFT 为 10064.405 ms，mean E2E 为 10390.638 ms。
  相比原 trace，本次同时提高了 token budget、选择了 Triton backend，并调整了
  GPU memory utilization；不能把时间差全部归因于 checkpoint 功能。
- 参考 `test.sh acc` 完成全部 164 道 HumanEval，`temperature=0`、`seed=0`、
  并发 32、`max_tokens=16384`，得到 160/164（97.56%），请求错误数为 0。
  用户确认 97% 已满足验收标准，额外的关闭功能对照已停止。
- 补充的 cached / cold 严格输出对照没有通过：四个 P rank 合计命中
  32512 个 prefix token，但随机 token、裸文本和 chat 文本的生成 token 存在
  差异，冷请求之间也观察到数值或输出变化。完整响应保留在验证目录的
  `prefix_*state_comparison.json`。关闭功能的对照未完成，因此尚不能归因，
  不据此声明整模型 cached / cold 逐 token 一致。
- 既有 worker dispatcher 回归的 8 个 fresh-process 测试被部署源码缺失
  `vllm/_version.py` 阻断：版本回退为 `dev`，在测试的版本检查阶段失败。
  没有为此修改 vLLM 本体。

验证产物位于
`/jsyadmin/guanyu/qwen3_8-next-flash-PD/2p1d-agent-loop/prof/runs/checkpoint_validation_20261118/`。
原始八份 trace 的路径见相邻
`checkpoint_20261118_int64_38400_util093/manifest.txt`，逐 forward 的分析见
验证目录内的 `prefill_trace_summary.json`。
