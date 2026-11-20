# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Skip local Qwen4Exp attention during online prefill-worker EP dummy runs.

Keep the existing nonempty dummy shape, PLE/HC, router, dispatch, experts and
combine. Only the local attention modules are bypassed. The worker entry point
owns the scope: startup profiling, graph capture and real requests never set it.
"""

from __future__ import annotations

import functools
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Iterator

import torch

from vllm_hcu.platforms import envs as henvs

_ACTIVE: ContextVar[bool] = ContextVar("hcu_prefill_ep_idle", default=False)
_BOUND = "_vllm_hcu_prefill_ep_idle_bound"
_SYNC_BOUND = "_vllm_hcu_prefill_ep_idle_sync_bound"


def supported_config(config: object) -> bool:
    """The first version supports the TP1 Qwen4Exp AgRs prefill topology."""
    if not henvs.VLLM_HCU_PREFILL_EP_IDLE:
        return False
    transfer = getattr(config, "kv_transfer_config", None)
    if getattr(transfer, "kv_role", None) != "kv_producer":
        return False
    model = getattr(config, "model_config", None)
    hf = getattr(model, "hf_text_config", None)
    if getattr(hf, "model_type", None) not in ("qwen4_exp", "qwen4_exp_text"):
        return False
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None and getattr(speculative, "method", None) != "mtp":
        return False
    parallel = getattr(config, "parallel_config", None)
    return (
        getattr(parallel, "enable_expert_parallel", False)
        and getattr(parallel, "data_parallel_size", 1) > 1
        and getattr(parallel, "all2all_backend", None) == "allgather_reducescatter"
        and not getattr(parallel, "enable_dbo", False)
        and all(
            getattr(parallel, field, 1) == 1
            for field in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
    )


def _bind_attention(attention: torch.nn.Module) -> bool:
    if getattr(attention, _BOUND, False):
        return False
    original = attention.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        # Real/capture compiled graphs always retain the original attention.
        # Online idle calls bypass compiled models through _EagerModel below.
        if not torch.compiler.is_compiling() and _ACTIVE.get():
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0]
            return torch.zeros_like(hidden)
        return original(*args, **kwargs)

    attention.forward = forward
    setattr(attention, _BOUND, True)
    return True


def install(runner: object) -> int:
    """Bind target and MTP attention after loading, before graph capture."""
    if not supported_config(runner.vllm_config):
        runner._hcu_prefill_ep_idle_enabled = False
        return 0
    count = 0
    for model in (runner.model, runner.get_draft_model()):
        if model is None:
            continue
        for layer in model.modules():
            cls = type(layer)
            if (
                cls.__name__ != "Qwen4ExpDecoderLayer"
                or not cls.__module__.startswith("vllm.models.qwen4_exp.")
            ):
                continue
            attribute = (
                "linear_attn" if layer.layer_type == "linear_attention" else "self_attn"
            )
            attention = getattr(layer, attribute)
            count += int(_bind_attention(attention))
    # An opt-in on the supported model must not silently become a no-op after
    # a model-layout change. Reinstallation is allowed for already-bound models.
    if not any(getattr(module, _BOUND, False) for module in runner.model.modules()):
        raise RuntimeError("prefill EP idle: no supported Qwen4Exp attention found")
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        _preserve_mtp_eager_sync(speculator)
    runner._hcu_prefill_ep_idle_enabled = True
    return count


def _preserve_mtp_eager_sync(speculator: object) -> None:
    """Keep the shared eager vote when MTP derives its uniform decode batch.

    Upstream can reset eager=False because a uniform draft batch can normally
    use a graph even after an eager target prefill. Here that would let the real
    rank replay attention while the idle rank skips it. Bind on *every* enabled
    rank, not only inside online_idle_scope; target_dp_sync is already global.
    """
    if getattr(speculator, _SYNC_BOUND, False):
        return
    original = speculator._build_uniform_batch_dp_sync

    @functools.wraps(original)
    def build_sync(target_dp_sync, *args, **kwargs):
        sync, num_tokens = original(target_dp_sync, *args, **kwargs)
        if target_dp_sync.eager:
            sync = replace(sync, eager=True)
        return sync, num_tokens

    speculator._build_uniform_batch_dp_sync = build_sync
    setattr(speculator, _SYNC_BOUND, True)


class _EagerGraphManager:
    """Vote NONE through the existing DP shape agreement; add no collective."""

    def __init__(self, original: object):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)

    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
        num_ubatches=1,
    ):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform_token_count=uniform_token_count,
            max_query_len=max_query_len,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )


class _EagerModel:
    """Bypass cached compiled code only while calling this idle rank's model."""

    def __init__(self, original: object):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __call__(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        previous = context.skip_compiled
        context.skip_compiled = True
        try:
            return self._original(*args, **kwargs)
        finally:
            context.skip_compiled = previous


@contextmanager
def online_idle_scope(runner: object) -> Iterator[None]:
    """Called only by Worker.execute_dummy_batch, never by general _dummy_run."""
    if not getattr(runner, "_hcu_prefill_ep_idle_enabled", False):
        yield
        return
    if _ACTIVE.get():
        yield
        return
    with ExitStack() as stack:
        token = _ACTIVE.set(True)
        stack.callback(_ACTIVE.reset, token)
        speculator = getattr(runner, "speculator", None)
        for owner in (runner, speculator):
            if owner is None:
                continue
            model = getattr(owner, "model", None)
            if model is not None:
                stack.callback(setattr, owner, "model", model)
                owner.model = _EagerModel(model)
            # MTP reuses the target DP agreement (including its eager vote).
            # Keep every draft manager eager as well if it does its own dispatch.
            for attribute in (
                "cudagraph_manager",
                "prefill_cudagraph_manager",
                "decode_cudagraph_manager",
            ):
                manager = getattr(owner, attribute, None)
                if manager is not None:
                    stack.callback(setattr, owner, attribute, manager)
                    setattr(owner, attribute, _EagerGraphManager(manager))
        yield
