# P 端 EP 空闲轮跳过 attention/GDN（实验开关）

当一个 DP rank 有真实请求、另一个没有时，后者仍须参加每层 MoE 的通信，
并计算分配给本地专家的对端真实 tokens。此功能只跳过空闲 rank 本地的
attention/GDN；不修改请求分配，不省略 MoE。

## 启用与范围

在 **P 服务的所有 DP worker** 启动环境中设置：

```bash
export VLLM_HCU_PREFILL_EP_IDLE=1
```

默认关闭；`VLLM_HCU_PREFILL_EP_IDLE=0` 可恢复原路径。
受总开关 `VLLM_HCU_USE_CUSTOM_OPS` 控制。需使用包含本改动的插件并启动新进程；
只修改运行中服务的 shell 环境不会生效。

第一版只在以下配置同时满足时启用，其他配置沿用原路径：

- Model Runner V2、Qwen4Exp（`qwen4_exp` / `qwen4_exp_text` 配置）；
- KV transfer role 为 `kv_producer`，不包含 D 端或 `kv_both`；
- DP > 1 且启用 EP，`all2all_backend=allgather_reducescatter`；
- TP、PP、PCP、DCP 均为 1，关闭 DBO；
- 不使用 speculative decoding，或使用 `method=mtp`。

该范围对应本次 TP1/DP2/EP2 的 P 端场景。开启参数不代表其他模型/拓扑也已支持。
成功绑定时 runner 会记录 `Prefill EP idle enabled`，包含 target/MTP attention 数量。

## 执行行为

空闲标识来自在线 `Worker.execute_dummy_batch`，不依据 token ID、激活是否为零，
也不依据一般性的 `dummy_run=True`。真实请求、启动内存 profiling、编译和 graph
capture 均保留原 attention 路径。

空闲作用域内，target 和 MTP 的 decoder attention/GDN 返回同形状的零输出，
保留 embedding、PLE、HC、MoE gate、dispatch、专家计算和 combine。
现有 dummy token 数、padding、attention metadata 及 MTP 步数保持不变；
例如 MTP 配置为 3 时仍使用原有的 4 个 target dummy tokens。
这里没有实现零 token batch，也没有将 embedding 清零。

已有 graph 含 attention 节点，直接 replay 无法跳过。因此空闲 rank 通过原有
DP batch 协商投票使用 `CUDAGraphMode.NONE`，这一轮所有 DP rank 均按协商进入
eager 模式，不增加 collective。仅空闲 rank 绕过模型的已编译代码以执行跳过分支；
有真实请求的 rank 仍执行完整模型。MTP 后续 uniform batch 继承 target 的全局
eager 结果，防止不同 rank 对 graph replay 的选择不一致。

这一版对 MTP 采取保守处理：**开启后，任何 target eager batch 的后续 MTP 也保持
eager，包括双方都有真实请求的情况**。现有 DP 协商没有单独传递“是否因本功能
而 eager”的标识，不能仅靠 target 的 eager 字段精确区分原因。这可能减少正常
prefill 后 MTP 的 graph 使用，A/B 必须覆盖双方繁忙的场景。

## 与 router 的关系

DP 请求 router 的策略、rank 选择和 scheduler 不变；HTTP header 指定 DP rank
的方式也不变。模型内 MoE gate 继续执行，真实 tokens 的激活和路由输入保留，
空闲 rank 仍处理对端发来的真实 tokens。

dummy 的 attention 输出改变后，**dummy 自身的专家选择可能改变**，其相关专家
计数和 EPLB 统计也可能变化。当前实现保留这些 dummy 的 gate/通信，不能解释成
消除了全部 dummy MoE 开销。实机仍需检查数值误差与路由后端的行为。

## 验证与收益边界

CPU 合同测试位于 `tests/runtime_patch/test_prefill_ep_idle.py`，覆盖开关/范围、
真实零值输入、attention 编译分支、异常恢复、target/MTP 及 eager 协商。
两进程 Gloo 测试使用简化的两专家 MoE 与真实 CPU collectives，比较开关前后
真实 rank 的路由/输出，验证空闲 rank 仍计算对端 tokens，且所有 target/MTP
层的通信次序一致。这些测试不替代真实 Qwen4Exp/HCU 精度和性能测试。

在插件仓库根目录运行，`/path/to/vllm-root` 指向匹配的 vLLM 源码目录：

```bash
PYTHONPATH=/path/to/vllm-root VLLM_PLUGINS=__disabled__ \
  python -m pytest -q tests/runtime_patch/test_prefill_ep_idle.py
```

建议在相同请求序列、预热状态下对 P 端做开关 A/B，覆盖：两边都有请求、
单边空闲、两边空闲、请求错峰以及 MTP 多步。比较输出、TTFT、P 吞吐与逐层耗时，
并检查 hang、NaN 和后续真实请求的状态。默认 `dp_sync_interval` 导致的尾部全空闲
轮与本功能独立，可另外对比 `--dp-sync-interval 1`，避免混淆收益来源。

该优化减少本地无效 attention/GDN 计算，不能消除真实 rank 的 prefill 计算或
修复请求到达不均衡。空闲 rank 可能更早进入 NCCL 等待，因此 profile 中 NCCL
事件变长也可能是预期现象。原本可 replay graph 的空闲轮改为 eager 后会增加
CPU 提交开销；全空闲尾部尤其需要实测，不能承诺端到端加速。
