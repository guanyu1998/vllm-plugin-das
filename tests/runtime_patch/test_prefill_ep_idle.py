# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts, including real two-rank Gloo collectives (no HCU kernels)."""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest
import torch
from torch import nn

from vllm_hcu.v1 import prefill_ep_idle as idle


def _config():
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen4_exp_text")
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            all2all_backend="allgather_reducescatter",
            enable_dbo=False,
        ),
        speculative_config=SimpleNamespace(method="mtp"),
    )


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return hidden_states * 0.5


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, kind="linear_attention", moe=None):
        super().__init__()
        self.layer_type = kind
        if kind == "linear_attention":
            self.linear_attn = _Attention()
        else:
            self.self_attn = _Attention()
        self.mlp = moe or nn.Identity()

    def forward(self, x):
        attention = (
            self.linear_attn if self.layer_type == "linear_attention" else self.self_attn
        )
        return self.mlp(x + attention(hidden_states=x))


Qwen4ExpDecoderLayer.__module__ = "vllm.models.qwen4_exp.amd.model"


class _Model(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = nn.ModuleList(layers or [Qwen4ExpDecoderLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _runner(draft=None):
    # Exercise upstream's derivation, which may reset eager=False.
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    manager = SimpleNamespace(sentinel="original manager")
    speculator = None
    if draft is not None:
        speculator = SimpleNamespace(
            model=draft,
            prefill_cudagraph_manager=manager,
            decode_cudagraph_manager=manager,
            _build_uniform_batch_dp_sync=lambda *args, **kwargs: (
                DraftModelSpeculator._build_uniform_batch_dp_sync(
                    None, *args, **kwargs
                )
            ),
        )
    return SimpleNamespace(
        model=_Model([Qwen4ExpDecoderLayer(), Qwen4ExpDecoderLayer("full_attention")]),
        vllm_config=_config(),
        speculator=speculator,
        cudagraph_manager=manager,
        get_draft_model=lambda: draft,
    )


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_PREFILL_EP_IDLE", "1")
    import vllm.forward_context as fc

    context = SimpleNamespace(skip_compiled=False)
    monkeypatch.setattr(fc, "get_forward_context", lambda: context)
    return context


def test_explicit_flag_and_master_switch(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_PREFILL_EP_IDLE", raising=False)
    assert not idle.supported_config(_config())
    monkeypatch.setenv("VLLM_HCU_PREFILL_EP_IDLE", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert idle.supported_config(_config())
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert not idle.supported_config(_config())


@pytest.mark.parametrize("role", [None, "kv_consumer", "kv_both"])
def test_only_dedicated_prefill_producer(enabled, role):
    config = _config()
    config.kv_transfer_config.kv_role = role
    assert not idle.supported_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("data_parallel_size", 1),
        ("enable_expert_parallel", False),
        ("all2all_backend", "deepep_high_throughput"),
        ("enable_dbo", True),
    ],
)
def test_unsupported_parallel_configs_do_not_activate(enabled, field, value):
    config = _config()
    setattr(config.parallel_config, field, value)
    assert not idle.supported_config(config)


def test_unsupported_model_and_speculation_do_not_activate(enabled):
    config = _config()
    config.model_config.hf_text_config.model_type = "qwen3_next"
    assert not idle.supported_config(config)
    config = _config()
    config.speculative_config.method = "eagle3"
    assert not idle.supported_config(config)
    config.speculative_config = None
    assert idle.supported_config(config)


def test_zero_values_do_not_mark_real_or_warmup_calls_idle(enabled):
    runner = _runner()
    assert idle.install(runner) == 2
    assert idle.install(runner) == 0
    attention = runner.model.layers[0].linear_attn
    runner.model(torch.zeros(4, 3))
    assert attention.calls == 1
    x = torch.ones(4, 3)
    expected = runner.model(x)
    with idle.online_idle_scope(runner):
        torch.testing.assert_close(runner.model(x), x)
    torch.testing.assert_close(runner.model(x), expected)
    assert attention.calls == 3


def test_target_draft_and_managers_restore_on_nested_scope_and_exception(enabled):
    runner = _runner(_Model())
    idle.install(runner)
    target, draft = runner.model, runner.speculator.model
    manager = runner.cudagraph_manager
    with pytest.raises(RuntimeError, match="model failure"):
        with idle.online_idle_scope(runner):
            proxy = runner.model
            with idle.online_idle_scope(runner):
                assert runner.model is proxy
                assert runner.cudagraph_manager.sentinel == "original manager"
                for model in (runner.model, runner.speculator.model):
                    torch.testing.assert_close(model(torch.ones(4, 2)), torch.ones(4, 2))
            assert enabled.skip_compiled is False
            raise RuntimeError("model failure")
    assert runner.model is target
    assert runner.speculator.model is draft
    assert runner.cudagraph_manager is manager
    assert runner.speculator.prefill_cudagraph_manager is manager
    assert runner.speculator.decode_cudagraph_manager is manager
    assert not idle._ACTIVE.get()
    assert enabled.skip_compiled is False


def test_skip_compiled_flag_is_scoped_even_if_forward_raises(enabled):
    def fail():
        assert enabled.skip_compiled is True
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        idle._EagerModel(fail)()
    assert enabled.skip_compiled is False
    enabled.skip_compiled = True
    with pytest.raises(RuntimeError, match="forward failed"):
        idle._EagerModel(fail)()
    assert enabled.skip_compiled is True


def test_disabled_install_leaves_model_and_dummy_scope_untouched(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_PREFILL_EP_IDLE", "0")
    runner = _runner()
    model = runner.model
    assert idle.install(runner) == 0
    with idle.online_idle_scope(runner):
        assert runner.model is model
        assert not idle._ACTIVE.get()
        torch.testing.assert_close(model(torch.ones(4, 2)), torch.full((4, 2), 2.25))


def test_worker_marks_only_online_dummy_entry_not_general_dummy_run(enabled):
    from vllm_hcu.v1.worker import HcuGPUWorker

    runner = _runner(_Model())
    idle.install(runner)
    runner.uniform_decode_query_len = 4
    observed = []

    def dummy_run(num_tokens, *, uniform_decode):
        assert num_tokens == 4 and uniform_decode
        observed.append(idle._ACTIVE.get())
        runner.model(torch.ones(4, 2))
        runner.speculator.model(torch.ones(4, 2))

    runner._dummy_run = dummy_run
    worker = HcuGPUWorker.__new__(HcuGPUWorker)
    worker.model_runner = runner
    worker.execute_dummy_batch()
    assert runner.model.layers[0].linear_attn.calls == 0
    assert runner.speculator.model.layers[0].linear_attn.calls == 0
    # Startup profiling/warmup calls _dummy_run directly, with no idle scope.
    runner._dummy_run(4, uniform_decode=True)
    assert observed == [True, False]
    assert runner.model.layers[0].linear_attn.calls == 1
    assert runner.speculator.model.layers[0].linear_attn.calls == 1


def test_enabled_unknown_model_layout_fails_closed(enabled):
    runner = _runner()
    runner.model = nn.Identity()
    with pytest.raises(RuntimeError, match="no supported Qwen4Exp attention"):
        idle.install(runner)


def test_compilation_captures_original_attention_not_idle_branch(enabled):
    attention = _Attention()
    idle._bind_attention(attention)
    compiled = torch.compile(attention, backend="eager", fullgraph=True)
    # Even compiling inside an idle scope must retain the regular attention.
    token = idle._ACTIVE.set(True)
    try:
        torch.testing.assert_close(compiled(torch.ones(4, 2)), torch.full((4, 2), 0.5))
        torch.testing.assert_close(attention(torch.ones(4, 2)), torch.zeros(4, 2))
    finally:
        idle._ACTIVE.reset(token)
        torch._dynamo.reset()


@pytest.mark.parametrize("eager", [False, True])
def test_mtp_uniform_decode_preserves_global_eager_vote(enabled, eager):
    from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
    from vllm.config.compilation import CUDAGraphMode

    runner = _runner(_Model())
    idle.install(runner)
    target_sync = DPSyncState(torch.tensor([9, 4]), None, eager=eager, num_reqs=3)
    sync, tokens = runner.speculator._build_uniform_batch_dp_sync(target_sync, 1)
    assert sync.eager is eager
    assert tokens == 3
    assert sync.num_tokens_across_dp.tolist() == [3, 3]
    if eager:
        # The REAL rank has no idle scope/manager proxy. It must also stay eager.
        desc, reused = dispatch_cg_and_sync_dp(
            None, 3, tokens, 1, dp_size=2, dp_rank=0, dp_sync=sync
        )
        assert desc.cg_mode == CUDAGraphMode.NONE
        assert reused is sync


class _GraphManager:
    def __init__(self, eager=False):
        self.eager = eager

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, **kwargs):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        return BatchExecutionDescriptor(
            CUDAGraphMode.NONE if self.eager else CUDAGraphMode.FULL,
            num_tokens, num_reqs,
            uniform_token_count=uniform_token_count,
        )


class _DistributedMoE(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.counts = None
        self.routes = []
        self.events = []
        self.peer_tokens_served = 0

    def forward(self, x):
        import torch.distributed as dist

        self.events.append("gate")
        self.routes.append((x[:, 0] >= 0).tolist())
        self.events.append("dispatch")
        padded = torch.zeros(max(self.counts), x.shape[1], dtype=x.dtype)
        padded[:x.shape[0]] = x
        gathered = [torch.empty_like(padded) for _ in self.counts]
        dist.all_gather(gathered, padded)
        all_x = torch.cat([part[:n] for part, n in zip(gathered, self.counts)])
        owners = (all_x[:, 0] >= 0).long()
        self.events.append("experts")
        if self.rank == 1:
            self.peer_tokens_served += int((owners[:self.counts[0]] == 1).sum())
        out = torch.where((owners == self.rank)[:, None], all_x * (self.rank + 2), 0)
        self.events.append("combine")
        dist.all_reduce(out)
        start = sum(self.counts[:self.rank])
        return out[start:start + x.shape[0]]


def _distributed_worker(rank, init_path, result_dir):
    import torch.distributed as dist
    import vllm.forward_context as fc
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import dp_utils

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=2,
        timeout=timedelta(seconds=25),
    )
    # Route upstream DP agreement through this real CPU process group.
    dp_utils.get_dp_group = lambda: SimpleNamespace(cpu_group=dist.group.WORLD)
    fc.get_forward_context = lambda: SimpleNamespace(skip_compiled=False)
    try:
        results = []
        for optimize in (False, True):
            runner = _runner(_Model([Qwen4ExpDecoderLayer(moe=_DistributedMoE(rank))]))
            runner.model = _Model([
                Qwen4ExpDecoderLayer(moe=_DistributedMoE(rank)),
                Qwen4ExpDecoderLayer("full_attention", moe=_DistributedMoE(rank)),
            ])
            # Reference uses normal attention in eager mode, so unequal real /
            # dummy lengths need no graph padding. Optimization must reach the
            # same global mode even though the real rank votes FULL.
            runner.cudagraph_manager = _GraphManager(eager=not optimize)
            runner.speculator.prefill_cudagraph_manager = _GraphManager()
            runner.speculator.decode_cudagraph_manager = _GraphManager()
            if optimize:
                idle.install(runner)
            target, draft = runner.model, runner.speculator.model
            outputs, modes = [], []
            with idle.online_idle_scope(runner) if rank == 1 else nullcontext():
                counts = [5, 4]
                desc, sync = dp_utils.dispatch_cg_and_sync_dp(
                    runner.cudagraph_manager, rank + 1, counts[rank], None, 2, rank
                )
                modes.append(desc.cg_mode.name)
                assert sync.num_tokens_across_dp.tolist() == counts
                for layer in target.layers:
                    layer.mlp.counts = counts
                x = torch.arange(counts[rank], dtype=torch.float64)[:, None] - 2
                outputs.append(runner.model(x).tolist())
                for step in range(3):
                    if step == 0:
                        manager = runner.speculator.prefill_cudagraph_manager
                        draft_sync, tokens, uniform = sync, counts[rank], None
                    else:
                        manager = runner.speculator.decode_cudagraph_manager
                        draft_sync, tokens = runner.speculator._build_uniform_batch_dp_sync(
                            sync, rank + 1
                        )
                        uniform = 1
                    desc, agreed = dp_utils.dispatch_cg_and_sync_dp(
                        manager, rank + 1, tokens, uniform, 2, rank, dp_sync=draft_sync
                    )
                    modes.append(desc.cg_mode.name)
                    draft.layers[0].mlp.counts = agreed.num_tokens_across_dp.tolist()
                    x = torch.arange(tokens, dtype=torch.float64)[:, None] - 2
                    outputs.append(runner.speculator.model(x).tolist())
                expected = ["NONE"] * 4 if optimize else ["NONE", "NONE", "FULL", "FULL"]
                assert modes == expected
            layers = [*target.layers, *draft.layers]
            results.append({
                "outputs": outputs,
                "routes": [layer.mlp.routes for layer in layers],
                "events": [layer.mlp.events for layer in layers],
                "attention_calls": sum(
                    (layer.linear_attn if layer.layer_type == "linear_attention"
                     else layer.self_attn).calls for layer in layers
                ),
                "peer_tokens": sum(layer.mlp.peer_tokens_served for layer in layers),
            })
        baseline, optimized = results
        assert baseline["events"] == optimized["events"]
        if rank == 0:
            assert baseline["outputs"] == optimized["outputs"]
            assert baseline["routes"] == optimized["routes"]
            assert optimized["attention_calls"] == baseline["attention_calls"] == 5
        else:
            assert baseline["attention_calls"] == 5
            assert optimized["attention_calls"] == 0
            assert optimized["peer_tokens"] > 0
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_two_rank_real_routes_outputs_and_target_mtp_collective_order(enabled, tmp_path):
    from torch.multiprocessing import spawn

    context = spawn(
        _distributed_worker, args=(str(tmp_path / "init"), str(tmp_path)),
        nprocs=2, join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank EP idle test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all((tmp_path / f"rank{rank}.json").is_file() for rank in range(2))
