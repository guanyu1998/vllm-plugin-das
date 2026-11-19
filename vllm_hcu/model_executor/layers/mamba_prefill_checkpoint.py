"""Internal prefill checkpoints for the v0.28.1 GDN and PLE cache API."""

from __future__ import annotations

import functools
import inspect
import os
from contextvars import ContextVar
from dataclasses import dataclass, field, make_dataclass, replace

import torch

from vllm.triton_utils import tl, triton


CHECKPOINT_ATTRIBUTE = "_hcu_prefill_checkpoint"
CAPABILITY_ATTRIBUTE = "_hcu_prefill_checkpoint_capability"
_MARKER = "_hcu_prefill_checkpoint_applied"
_ACTIVE_GDN: ContextVar[GDNCheckpointContext | None] = ContextVar(
    "hcu_gdn_prefill_checkpoint", default=None
)


def enabled() -> bool:
    return os.environ.get("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", "0").lower() in (
        "1", "true", "yes", "on"
    )


@dataclass
class PrefillCheckpoint:
    offsets: torch.Tensor
    state_indices: torch.Tensor
    offsets_cpu: list[int]


@dataclass
class GDNCheckpointContext:
    metadata: object
    recurrent_state: torch.Tensor
    conv_history_len: int
    exported: bool = False


def checkpoint_positions(
    seq_lens, query_lens, hash_block_size, block_size, alignment, drop_eagle_block
):
    from vllm.v1.kv_cache_interface import (
        get_mamba_prefill_checkpoint_position,
        is_mamba_prefill_checkpoint_valid,
    )

    offsets, columns = [], []
    for seq_len, query_len in zip(seq_lens, query_lens):
        start = seq_len - query_len
        position = get_mamba_prefill_checkpoint_position(
            seq_len, hash_block_size, drop_eagle_block
        )
        valid = is_mamba_prefill_checkpoint_valid(
            start, seq_len, position, hash_block_size, block_size, alignment
        )
        offsets.append(position - start if valid else 0)
        columns.append((seq_len + block_size - 1) // block_size - 2 if valid else -1)
    return offsets, columns


def build_checkpoint(builder, common, rows) -> PrefillCheckpoint | None:
    spec = builder.kv_cache_spec
    if spec.num_prefill_checkpoint_blocks == 0:
        return None
    config = builder.vllm_config
    query_lens = common.query_start_loc_cpu.diff().tolist()
    seq_lens = common.seq_lens_cpu_upper_bound
    if seq_lens is None:
        raise RuntimeError("Internal Mamba checkpoints require CPU sequence lengths")
    speculative = config.speculative_config
    offsets, columns = checkpoint_positions(
        [int(seq_lens[row]) for row in rows],
        [query_lens[row] for row in rows],
        config.cache_config.prefix_match_unit or spec.block_size,
        spec.block_size,
        spec.prefill_checkpoint_alignment,
        speculative is not None and speculative.use_eagle_block_drop(),
    )
    if not any(offsets):
        return None
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    device = common.block_table_tensor.device
    row_tensor = async_tensor_h2d(rows, device=device, dtype=torch.int64)
    col_tensor = async_tensor_h2d(columns, device=device, dtype=torch.int64)
    indices = common.block_table_tensor[row_tensor, col_tensor.clamp_min(0)]
    indices = torch.where(col_tensor >= 0, indices, NULL_BLOCK_ID)
    return PrefillCheckpoint(
        async_tensor_h2d(offsets, device=device, dtype=torch.int32),
        indices,
        offsets,
    )


def patch_metadata_builder(builder_cls) -> bool:
    if vars(builder_cls).get(_MARKER, False):
        return False
    original = builder_cls.build
    signature = inspect.signature(original)

    @functools.wraps(original)
    def build(self, *args, **kwargs):
        metadata = original(self, *args, **kwargs)
        checkpoint = None
        if self.kv_cache_spec.num_prefill_checkpoint_blocks and metadata.num_prefills:
            arguments = signature.bind(self, *args, **kwargs).arguments
            common = arguments["common_attn_metadata"]
            rows = list(range(common.num_reqs))
            drafts = arguments.get("num_decode_draft_tokens_cpu")
            if self.use_spec_decode and drafts is not None:
                rows = [row for row in rows if int(drafts[row]) < 0]
            checkpoint = build_checkpoint(self, common, rows)
        setattr(metadata, CHECKPOINT_ATTRIBUTE, checkpoint)
        return metadata

    builder_cls.build = build
    setattr(builder_cls, _MARKER, True)
    return True


def patch_metadata_type(module, builder_cls, metadata_name) -> None:
    original = getattr(module, metadata_name)
    if CHECKPOINT_ATTRIBUTE in original.__dataclass_fields__:
        return
    metadata_cls = make_dataclass(
        metadata_name,
        [(CHECKPOINT_ATTRIBUTE, PrefillCheckpoint | None, field(default=None))],
        bases=(original,), namespace={"__module__": module.__name__}, kw_only=True,
    )
    setattr(module, metadata_name, metadata_cls)
    if getattr(builder_cls, "metadata_cls", None) is original:
        builder_cls.metadata_cls = metadata_cls


def patch_layer_spec(layer_cls, kind: str) -> None:
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    original = layer_cls.get_kv_cache_spec

    def capable(self, config):
        if not enabled() or config.cache_config.mamba_cache_mode != "align":
            return False
        if kind == "gdn":
            from vllm_hcu.patch.worker.op_opt._gdn_backend import (
                requested_prefill_backend,
            )

            return (
                self.gdn_prefill_backend == "triton"
                and requested_prefill_backend(config) == "triton"
                and self.conv_kernel_size - 1 <= 64
            )
        return self.conv_state_len <= 64

    @functools.wraps(original)
    def get_spec(self, config):
        spec = original(self, config)
        if spec is None or not capable(self, config):
            return spec
        from vllm.config import get_layers_from_vllm_config

        layers = get_layers_from_vllm_config(config, MambaBase).values()
        if not all(
            getattr(layer, CAPABILITY_ATTRIBUTE, lambda _: False)(config)
            for layer in layers
        ):
            return spec
        return replace(
            spec, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=64
        )

    setattr(layer_cls, CAPABILITY_ATTRIBUTE, capable)
    layer_cls.get_kv_cache_spec = get_spec


@triton.jit
def _store_conv_checkpoint(
    inputs, state, starts, offsets, destinations, token_indices,
    input_stride_token: tl.constexpr, input_stride_channel: tl.constexpr,
    state_stride_slot: tl.constexpr, state_stride_channel: tl.constexpr,
    state_stride_window: tl.constexpr, channels: tl.constexpr,
    window: tl.constexpr, map_tokens: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.load(offsets + row)
    destination = tl.load(destinations + row).to(tl.int64)
    elements = tl.program_id(1) * block + tl.arange(0, block)
    channel = elements // window
    history = elements % window
    valid = (elements < channels * window) & (offset >= window) & (offset > 0)
    token = tl.load(starts + row).to(tl.int64) + offset - window + history
    if map_tokens:
        token = tl.load(token_indices + token, mask=valid, other=0).to(tl.int64)
    values = tl.load(
        inputs + token * input_stride_token + channel * input_stride_channel,
        mask=valid, other=0,
    )
    tl.store(
        state + destination * state_stride_slot + channel * state_stride_channel
        + history * state_stride_window, values, mask=valid,
    )


def export_conv(checkpoint, inputs, state, starts, window, token_indices=None):
    if window == 0:
        return
    if (
        inputs.ndim != 2 or state.ndim != 3
        or inputs.shape[1] != state.shape[1]
        or not 0 < window <= state.shape[2]
    ):
        raise RuntimeError("Checkpoint conv history does not match the cache layout")
    if any(0 < offset < window for offset in checkpoint.offsets_cpu):
        raise RuntimeError("Checkpoint offset does not contain the full conv history")
    _store_conv_checkpoint[(len(checkpoint.offsets_cpu), triton.cdiv(inputs.shape[1] * window, 512))](
        inputs, state, starts, checkpoint.offsets, checkpoint.state_indices,
        token_indices if token_indices is not None else starts,
        *inputs.stride(), *state.stride(), inputs.shape[1], window,
        token_indices is not None, 512,
    )


@triton.jit
def _store_recurrent_checkpoint(
    snapshots, state, offsets, destinations, row_size: tl.constexpr,
    state_stride: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    valid_row = tl.load(offsets + row) > 0
    destination = tl.load(destinations + row).to(tl.int64)
    elements = tl.program_id(1) * block + tl.arange(0, block)
    mask = valid_row & (elements < row_size)
    values = tl.load(snapshots + row.to(tl.int64) * row_size + elements, mask=mask, other=0)
    tl.store(state + destination * state_stride + elements, values, mask=mask)


def checkpoint_chunk(*, checkpoint, state, q, k, v, g, beta,
                     initial_state, output_final_state, cu_seqlens=None,
                     chunk_indices=None, chunk_offsets=None,
                     use_qk_l2norm_in_kernel=False, core_attn_out=None, scale=None):
    from vllm.third_party.flash_linear_attention.ops import chunk as chunk_ops
    from vllm_hcu.model_executor.layers.fla_prefill_checkpoint import (
        chunk_gated_delta_rule_fwd_h,
    )

    if use_qk_l2norm_in_kernel:
        q, k = chunk_ops.l2norm_fwd(q), chunk_ops.l2norm_fwd(k)
    q, k, v, g, beta = (tensor.contiguous() for tensor in (q, k, v, g, beta))
    g = chunk_ops.chunk_local_cumsum(
        g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    attention = chunk_ops.chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, output_dtype=torch.float32,
    )
    attention = chunk_ops.solve_tril(
        A=attention, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    weights, values = chunk_ops.recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=attention, g_cumsum=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    if tuple(state.stride()[1:]) != (state.shape[2] * state.shape[3], state.shape[3], 1):
        raise RuntimeError("Recurrent checkpoint cache rows must be contiguous")
    snapshots = torch.empty(initial_state.shape, dtype=state.dtype, device=k.device)
    hidden, new_values, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=weights, u=values, g=g, initial_state=initial_state.contiguous(),
        output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        checkpoint_state=snapshots, checkpoint_offsets=checkpoint.offsets,
    )
    row_size = snapshots[0].numel()
    _store_recurrent_checkpoint[(snapshots.shape[0], triton.cdiv(row_size, 512))](
        snapshots, state, checkpoint.offsets, checkpoint.state_indices,
        row_size, state.stride(0), 512,
    )
    output = chunk_ops.chunk_fwd_o(
        q=q, k=k, v=new_values, h=hidden, g=g,
        scale=k.shape[-1] ** -0.5 if scale is None else scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    return output.to(q.dtype), final_state


def patch_gdn(module) -> bool:
    layer_cls = module.QwenGatedDeltaNetAttention
    if vars(layer_cls).get(_MARKER, False):
        return False
    patch_layer_spec(layer_cls, "gdn")
    original_core = layer_cls._forward_core
    original_conv = module.causal_conv1d_fn
    original_chunk = module.fla_chunk_gated_delta_rule

    @functools.wraps(original_core)
    def core(self, *args, **kwargs):
        if not enabled():
            return original_core(self, *args, **kwargs)
        metadata = module.get_forward_context().attn_metadata
        metadata = metadata.get(self.prefix) if isinstance(metadata, dict) else None
        checkpoint = getattr(metadata, CHECKPOINT_ATTRIBUTE, None)
        if checkpoint is None:
            return original_core(self, *args, **kwargs)
        context = GDNCheckpointContext(
            metadata, self.kv_cache[1], self.conv_kernel_size - 1
        )
        token = _ACTIVE_GDN.set(context)
        try:
            result = original_core(self, *args, **kwargs)
            if not context.exported:
                raise RuntimeError("GDN cache reserved a checkpoint without exporting it")
            return result
        finally:
            _ACTIVE_GDN.reset(token)

    @functools.wraps(original_conv)
    def conv(inputs, *args, **kwargs):
        context = _ACTIVE_GDN.get()
        if context is not None:
            metadata = context.metadata
            export_conv(
                getattr(metadata, CHECKPOINT_ATTRIBUTE), inputs.transpose(0, 1),
                kwargs["conv_states"], metadata.non_spec_query_start_loc,
                context.conv_history_len,
            )
        return original_conv(inputs, *args, **kwargs)

    @functools.wraps(original_chunk)
    def chunk(*args, **kwargs):
        context = _ACTIVE_GDN.get()
        if context is None:
            return original_chunk(*args, **kwargs)
        if args:
            raise RuntimeError("Audited GDN checkpoint call requires named inputs")
        metadata = context.metadata
        checkpoint = getattr(metadata, CHECKPOINT_ATTRIBUTE)
        first = len(checkpoint.offsets_cpu) - len(kwargs["initial_state"])
        prefill = PrefillCheckpoint(
            checkpoint.offsets[first:], checkpoint.state_indices[first:],
            checkpoint.offsets_cpu[first:],
        )
        result = checkpoint_chunk(
            checkpoint=prefill, state=context.recurrent_state, **kwargs,
        )
        context.exported = True
        return result

    layer_cls._forward_core = core
    module.causal_conv1d_fn = conv
    module.fla_chunk_gated_delta_rule = chunk
    setattr(layer_cls, _MARKER, True)
    return True


def patch_ple(module) -> bool:
    layer_cls = module.Qwen4ExpPLELayer
    if vars(layer_cls).get(_MARKER, False):
        return False
    patch_layer_spec(layer_cls, "ple")
    original = layer_cls._short_conv_dilated_dispatch
    signature = inspect.signature(original)

    @functools.wraps(original)
    def dispatch(self, *args, **kwargs):
        if not enabled():
            return original(self, *args, **kwargs)
        metadata = kwargs.get("metadata")
        arguments = kwargs
        if metadata is None:
            arguments = signature.bind(self, *args, **kwargs).arguments
            metadata = arguments["metadata"]
        checkpoint = getattr(metadata, CHECKPOINT_ATTRIBUTE, None)
        if checkpoint is not None:
            export_conv(
                checkpoint, arguments["inputs"], arguments["conv_state"],
                metadata.non_spec_query_start_loc, self.conv_state_len,
                token_indices=metadata.non_spec_token_indx,
            )
        return original(self, *args, **kwargs)

    layer_cls._short_conv_dilated_dispatch = dispatch
    setattr(layer_cls, _MARKER, True)
    return True
