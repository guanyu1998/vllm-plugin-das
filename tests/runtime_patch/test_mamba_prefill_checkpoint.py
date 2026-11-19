from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import mamba_prefill_checkpoint as checkpoints


def test_checkpoint_positions_follow_deployed_cache_rules():
    offsets, columns = checkpoints.checkpoint_positions(
        [256000, 256000, 256000, 512],
        [35264, 35199, 1, 512],
        hash_block_size=64, block_size=3200, alignment=64, drop_eagle_block=True,
    )
    assert offsets == [35136, 0, 0, 0]
    assert columns == [78, -1, -1, -1]


def test_layer_spec_requires_every_mamba_layer(monkeypatch):
    import vllm.config
    from vllm.v1.kv_cache_interface import MambaSpec

    class Layer:
        conv_state_len = 8

        def get_kv_cache_spec(self, config):
            return MambaSpec(
                block_size=3200, shapes=((8, 4),), dtypes=(torch.float32,),
                mamba_type="short_conv", mamba_cache_mode="align",
            )

    checkpoints.patch_layer_spec(Layer, "ple")
    layer = Layer()
    layers = {"ple": layer}
    monkeypatch.setattr(vllm.config, "get_layers_from_vllm_config", lambda *args: layers)
    config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    monkeypatch.delenv("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", raising=False)
    assert layer.get_kv_cache_spec(config).num_prefill_checkpoint_blocks == 0
    monkeypatch.setenv("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", "1")
    spec = layer.get_kv_cache_spec(config)
    assert spec.num_prefill_checkpoint_blocks == 1
    assert spec.prefill_checkpoint_alignment == 64
    layers["unsupported"] = object()
    assert layer.get_kv_cache_spec(config).num_prefill_checkpoint_blocks == 0


def test_metadata_uses_own_group_and_excludes_spec_rows(monkeypatch):
    from vllm.v1.kv_cache_interface import MambaSpec

    class Builder:
        use_spec_decode = True

        def build(self, common_prefix_len, common_attn_metadata,
                  num_decode_draft_tokens_cpu=None, fast_build=False):
            return SimpleNamespace(num_prefills=2)

    checkpoints.patch_metadata_builder(Builder)
    assert checkpoints.patch_metadata_builder(Builder) is False
    builder = Builder()
    builder.kv_cache_spec = MambaSpec(
        block_size=3200, shapes=((8, 4),), dtypes=(torch.float32,),
        mamba_type="short_conv", mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=64,
    )
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(prefix_match_unit=64), speculative_config=None
    )
    common = SimpleNamespace(
        num_reqs=3, query_start_loc_cpu=torch.tensor([0, 35264, 35267, 70531]),
        seq_lens_cpu_upper_bound=torch.tensor([256000, 128, 256000]),
        block_table_tensor=torch.arange(3 * 80).view(3, 80),
    )
    monkeypatch.setattr(
        "vllm.utils.torch_utils.async_tensor_h2d",
        lambda data, device, dtype: torch.tensor(data, device=device, dtype=dtype),
    )
    metadata = builder.build(0, common, torch.tensor([-1, 2, -1]))
    checkpoint = getattr(metadata, checkpoints.CHECKPOINT_ATTRIBUTE)
    assert checkpoint.offsets_cpu == [35200, 35200]
    assert checkpoint.state_indices.tolist() == [78, 238]
    common.block_table_tensor = common.block_table_tensor + 1000
    other = builder.build(0, common, torch.tensor([-1, 2, -1]))
    assert getattr(other, checkpoints.CHECKPOINT_ATTRIBUTE).state_indices.tolist() == [1078, 1238]


def test_gdn_checkpoint_context_is_reset_on_failure(monkeypatch):
    class Layer:
        conv_kernel_size = 4

        def get_kv_cache_spec(self, config):
            return None

        def _forward_core(self):
            assert checkpoints._ACTIVE_GDN.get() is not None
            raise ValueError("forward failed")

    metadata = SimpleNamespace()
    setattr(metadata, checkpoints.CHECKPOINT_ATTRIBUTE, object())
    module = SimpleNamespace(
        QwenGatedDeltaNetAttention=Layer,
        causal_conv1d_fn=lambda: None, fla_chunk_gated_delta_rule=lambda: None,
        get_forward_context=lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    assert checkpoints.patch_gdn(module)
    assert checkpoints.patch_gdn(module) is False
    layer = Layer()
    layer.prefix = "layer"
    layer.kv_cache = (None, None)
    monkeypatch.setenv("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", "1")
    with pytest.raises(ValueError, match="forward failed"):
        layer._forward_core()
    assert checkpoints._ACTIVE_GDN.get() is None


def test_checkpoint_survives_dataclass_metadata_replacement():
    from dataclasses import dataclass, replace

    @dataclass
    class Metadata:
        num_prefills: int

    class Builder:
        metadata_cls = Metadata

    module = SimpleNamespace(__name__="checkpoint_test_metadata", Metadata=Metadata)
    checkpoints.patch_metadata_type(module, Builder, "Metadata")
    checkpoint = object()
    metadata = module.Metadata(1, **{checkpoints.CHECKPOINT_ATTRIBUTE: checkpoint})
    copied = replace(metadata, num_prefills=2)
    assert isinstance(copied, Metadata)
    assert getattr(copied, checkpoints.CHECKPOINT_ATTRIBUTE) is checkpoint
    assert Builder.metadata_cls is module.Metadata


def test_gdn_rejects_a_reserved_checkpoint_without_export(monkeypatch):
    class Layer:
        conv_kernel_size = 4

        def get_kv_cache_spec(self, config):
            return None

        def _forward_core(self):
            return None

    metadata = SimpleNamespace()
    setattr(metadata, checkpoints.CHECKPOINT_ATTRIBUTE, object())
    module = SimpleNamespace(
        QwenGatedDeltaNetAttention=Layer,
        causal_conv1d_fn=lambda: None, fla_chunk_gated_delta_rule=lambda: None,
        get_forward_context=lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    checkpoints.patch_gdn(module)
    layer = Layer()
    layer.prefix = "layer"
    layer.kv_cache = (None, None)
    monkeypatch.setenv("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", "1")
    with pytest.raises(RuntimeError, match="without exporting"):
        layer._forward_core()
    assert checkpoints._ACTIVE_GDN.get() is None


def test_conv_checkpoint_rejects_weight_layout_as_history_length():
    with pytest.raises(RuntimeError, match="cache layout"):
        checkpoints.export_conv(
            None, torch.empty(128, 768), torch.empty(4, 768, 6),
            None, window=767,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
def test_triton_checkpoint_matches_stopped_forward_and_continuation(cache_dtype):
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

    torch.manual_seed(42)
    device = "cuda"
    length, offset, heads, head_dim = 300, 128, 2, 128
    query = torch.nn.functional.normalize(
        torch.randn(1, length, heads, head_dim, device=device), dim=-1
    ).bfloat16()
    key = torch.nn.functional.normalize(torch.randn_like(query.float()), dim=-1).bfloat16()
    value = torch.randn_like(query)
    gating = torch.nn.functional.logsigmoid(torch.randn(1, length, heads, device=device))
    beta = torch.sigmoid(torch.randn_like(gating))
    initial = torch.randn(1, heads, head_dim, head_dim, device=device) * 0.01
    state = torch.full((4, heads, head_dim, head_dim), 7.0, dtype=cache_dtype, device=device)
    checkpoint = checkpoints.PrefillCheckpoint(
        torch.tensor([offset], device=device, dtype=torch.int32),
        torch.tensor([2], device=device, dtype=torch.int32), [offset],
    )
    starts = torch.tensor([0, length], dtype=torch.int32, device=device)
    inputs = dict(q=query, k=key, v=value, g=gating, beta=beta)
    output, final = checkpoints.checkpoint_chunk(
        checkpoint=checkpoint, state=state, **inputs, initial_state=initial,
        output_final_state=True, cu_seqlens=starts,
    )
    reference_output, reference_final = chunk_gated_delta_rule(
        **inputs, initial_state=initial, output_final_state=True, cu_seqlens=starts
    )
    prefix_inputs = {name: tensor[:, :offset] for name, tensor in inputs.items()}
    _, stopped = chunk_gated_delta_rule(
        **prefix_inputs, initial_state=initial, output_final_state=True,
        cu_seqlens=torch.tensor([0, offset], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(output, reference_output, atol=0, rtol=0)
    torch.testing.assert_close(final, reference_final, atol=0, rtol=0)
    torch.testing.assert_close(state[2], stopped[0].to(cache_dtype), atol=0, rtol=0)
    assert (state[[0, 1, 3]] == 7).all()
    tail_inputs = {name: tensor[:, offset:] for name, tensor in inputs.items()}
    tail_output, tail_final = chunk_gated_delta_rule(
        **tail_inputs, initial_state=state[2:3].float(), output_final_state=True,
        cu_seqlens=torch.tensor([0, length - offset], device=device, dtype=torch.int32),
    )
    tolerance = 0 if cache_dtype == torch.float32 else 0.02
    torch.testing.assert_close(tail_output, output[:, offset:], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(tail_final, final, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_varlen_checkpoint_skips_invalid_rows_and_uses_relative_offsets():
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

    torch.manual_seed(43)
    lengths, offsets, slots = [300, 150, 250], [128, 0, 192], [3, 0, 4]
    starts_cpu = [0, 300, 450, 700]
    query = torch.nn.functional.normalize(
        torch.randn(1, sum(lengths), 1, 128, device="cuda"), dim=-1
    ).bfloat16()
    key = torch.nn.functional.normalize(torch.randn_like(query.float()), dim=-1).bfloat16()
    inputs = dict(
        q=query, k=key, v=torch.randn(1, sum(lengths), 3, 128, device="cuda", dtype=torch.bfloat16),
        g=torch.nn.functional.logsigmoid(torch.randn(1, sum(lengths), 3, device="cuda")),
        beta=torch.rand(1, sum(lengths), 3, device="cuda"),
    )
    initial = torch.randn(3, 3, 128, 128, device="cuda") * 0.01
    state = torch.full((6, 3, 128, 128), 9.0, device="cuda")
    checkpoint = checkpoints.PrefillCheckpoint(
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        torch.tensor(slots, device="cuda", dtype=torch.int32), offsets,
    )
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    output, final = checkpoints.checkpoint_chunk(
        checkpoint=checkpoint, state=state, **inputs, initial_state=initial,
        output_final_state=True, cu_seqlens=starts,
    )
    reference_output, reference_final = chunk_gated_delta_rule(
        **inputs, initial_state=initial, output_final_state=True, cu_seqlens=starts
    )
    torch.testing.assert_close(output, reference_output, atol=0, rtol=0)
    torch.testing.assert_close(final, reference_final, atol=0, rtol=0)
    for row in (0, 2):
        start, offset = starts_cpu[row], offsets[row]
        prefix_inputs = {name: tensor[:, start:start + offset] for name, tensor in inputs.items()}
        _, stopped = chunk_gated_delta_rule(
            **prefix_inputs, initial_state=initial[row:row + 1], output_final_state=True,
            cu_seqlens=torch.tensor([0, offset], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(state[slots[row]], stopped[0], atol=0, rtol=0)
    assert (state[[0, 1, 2, 5]] == 9).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_checkpoint_exports_address_large_cache_slots_with_int64(cache_dtype):
    slot_stride = 2**30 + 128
    width = 4 if cache_dtype == torch.float32 else 3
    storage = torch.empty(2 * slot_stride + width, dtype=cache_dtype, device="cuda")
    checkpoint = checkpoints.PrefillCheckpoint(
        torch.tensor([64], dtype=torch.int32, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"), [64],
    )
    if cache_dtype == torch.bfloat16:
        state = storage.as_strided((3, 1, width), (slot_stride, width, 1))
        state[0].fill_(7)
        state[2].fill_(9)
        inputs = torch.arange(128, dtype=cache_dtype, device="cuda").view(-1, 1)
        checkpoints.export_conv(
            checkpoint, inputs, state,
            torch.tensor([0, 128], dtype=torch.int32, device="cuda"), width,
        )
        torch.testing.assert_close(state[2], inputs[61:64].T, atol=0, rtol=0)
    else:
        state = storage.as_strided((3, 1, 1, width), (slot_stride, width, width, 1))
        state[0].fill_(7)
        state[2].fill_(9)
        snapshots = torch.tensor([2, 3, 5, 7], dtype=cache_dtype, device="cuda")
        checkpoints._store_recurrent_checkpoint[(1, 1)](
            snapshots, state, checkpoint.offsets, checkpoint.state_indices,
            width, slot_stride, 512,
        )
        torch.testing.assert_close(state[2].flatten(), snapshots, atol=0, rtol=0)
    assert (state[0] == 7).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_conv_checkpoint_has_correct_history_and_preserves_other_slots(mapped, transposed):
    device = "cuda"
    channels, window = 7, 5
    raw = torch.arange(30 * channels, dtype=torch.float32, device=device).view(30, channels)
    token_indices = torch.arange(29, -1, -1, device=device) if mapped else None
    starts = torch.tensor([0, 12, 30], device=device, dtype=torch.int32)
    state = torch.full((4, window + 3, channels), -7.0, device=device)
    state = state.transpose(1, 2) if transposed else state.transpose(1, 2).contiguous()
    checkpoint = checkpoints.PrefillCheckpoint(
        torch.tensor([8, 0], device=device, dtype=torch.int32),
        torch.tensor([2, 0], device=device, dtype=torch.int32), [8, 0],
    )
    checkpoints.export_conv(checkpoint, raw, state, starts, window, token_indices)
    expected = raw[3:8] if not mapped else raw[token_indices[3:8]]
    torch.testing.assert_close(state[2, :, :window], expected.T, atol=0, rtol=0)
    assert (state[[0, 1, 3]] == -7).all()
    assert (state[2, :, window:] == -7).all()


def test_scheduler_keeps_budget_limit_but_allows_last_forward():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=3200), hash_block_size=64,
        mamba_has_prefill_checkpoint_blocks=True, mamba_prefill_checkpoint_alignment=64,
        use_eagle_block_drop=True, mamba_partial_cache_hit=True,
        mamba_fine_grained_prefix_cache=False,
        max_num_scheduled_tokens=32000,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    request = SimpleNamespace(
        request_id="checkpoint-resume",
        num_computed_tokens=220736, num_prompt_tokens=256000, num_tokens=256000,
        shared_prefix_boundary=0,
    )
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 35264) == 35264
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 28800) == 25664
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 35264) == 64


@pytest.mark.parametrize("use_eagle", [False, True])
def test_cache_manager_reserves_and_reuses_checkpoint_for_both_groups(use_eagle):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec,
    )
    from vllm.v1.request import Request

    init_none_hash(sha256)
    specs = [FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=8, dtype=torch.float32)]
    for shape in [((4, 8), (2, 8, 8)), ((8, 8),)]:
        specs.append(MambaSpec(
            block_size=3200, shapes=shape, dtypes=(torch.float32,) * len(shape),
            mamba_cache_mode="align", num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=64,
        ))
    config = KVCacheConfig(
        num_blocks=1000, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer{index}"], spec) for index, spec in enumerate(specs)],
    )
    manager = KVCacheManager(
        config, max_model_len=10000, scheduler_block_size=3200,
        hash_block_size=64, enable_caching=True, use_eagle=use_eagle,
    )
    tokens = list(range(6410))

    def request(name):
        return Request(
            request_id=name, prompt_token_ids=tokens, sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None, block_hasher=get_request_block_hasher(64, sha256),
        )

    first = request("first")
    allocated = manager.allocate_slots(first, len(tokens), delay_cache_blocks=True)
    assert allocated is not None
    position = 6336 if use_eagle else 6400
    for single in manager.coordinator.single_type_managers[1:]:
        assert single._checkpoint_positions[first.request_id] == position
        block = single.req_to_blocks[first.request_id][1]
        assert not block.is_null
    manager.cache_blocks(first, len(tokens))
    for single in manager.coordinator.single_type_managers[1:]:
        assert single.req_to_blocks[first.request_id][1].block_hash_num_tokens == position
    _, hit, _ = manager.get_computed_blocks(request("second"))
    assert hit == position


def test_checkpoint_callbacks_are_registered_for_exact_modules():
    from vllm_hcu.patch.worker import worker_callback_names

    callbacks = dict(worker_callback_names())
    assert callbacks["worker.op_opt.mamba.gdn.prefill_checkpoint"] == "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    assert callbacks["worker.core_fix.qwen4_exp.ple.prefill_checkpoint"] == "vllm.models.qwen4_exp.amd.ple_layer"
    assert callbacks["worker.framework_opt.mamba.gdn.checkpoint_metadata"] == "vllm.v1.attention.backends.gdn_attn"
    assert callbacks["worker.framework_opt.qwen4_exp.ple.checkpoint_metadata"] == "vllm.v1.attention.backends.short_conv_attn"


def _restore_layer_patch(monkeypatch, module, layer_name, gdn=False):
    layer_cls = getattr(module, layer_name)
    for name in ("get_kv_cache_spec", checkpoints.CAPABILITY_ATTRIBUTE, checkpoints._MARKER):
        monkeypatch.setattr(layer_cls, name, getattr(layer_cls, name, None), raising=False)
    method = "_forward_core" if gdn else "_short_conv_dilated_dispatch"
    monkeypatch.setattr(layer_cls, method, getattr(layer_cls, method))
    if gdn:
        for name in ("causal_conv1d_fn", "fla_chunk_gated_delta_rule"):
            monkeypatch.setattr(module, name, getattr(module, name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("transposed_weight", [False, True])
def test_real_gdn_core_exports_both_states_and_resumes(monkeypatch, transposed_weight):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm_hcu.patch.worker.op_opt._gdn_common import normalize_nn_conv_weight

    _restore_layer_patch(monkeypatch, gdn, "QwenGatedDeltaNetAttention", gdn=True)
    original_conv = gdn.causal_conv1d_fn

    def normalized_conv(inputs, weight, *args, **kwargs):
        weight = normalize_nn_conv_weight(weight, inputs.shape[0], "test.conv")
        return original_conv(inputs, weight, *args, **kwargs)

    monkeypatch.setattr(gdn, "causal_conv1d_fn", normalized_conv)
    checkpoints.patch_gdn(gdn)
    monkeypatch.setenv("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", "1")
    torch.manual_seed(7)
    length, offset, heads, dimension, window = 300, 128, 2, 128, 3
    channels = 3 * heads * dimension
    raw = torch.randn(length, channels, device="cuda", dtype=torch.bfloat16)
    gating_a = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16)
    gating_b = torch.randn_like(gating_a)
    initial_conv = torch.randn(4, channels, window, device="cuda", dtype=torch.bfloat16)
    initial_recurrent = torch.randn(4, heads, dimension, dimension, device="cuda") * 0.01
    layer = SimpleNamespace(
        prefix="gdn", enable_packed_recurrent_decode=False, tp_size=1,
        num_k_heads=heads, num_v_heads=heads, head_k_dim=dimension, head_v_dim=dimension,
        conv_kernel_size=window + 1,
        key_dim=heads * dimension, value_dim=heads * dimension, activation="silu",
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.zeros(heads, device="cuda", dtype=torch.bfloat16),
        conv1d=SimpleNamespace(
            weight=torch.randn(channels, 1, window + 1, device="cuda", dtype=torch.bfloat16) * 0.1,
            bias=None,
        ),
        chunk_gated_delta_rule=lambda **kwargs: gdn.fla_chunk_gated_delta_rule(**kwargs),
    )
    if transposed_weight:
        layer.conv1d.weight = layer.conv1d.weight.transpose(0, 2).contiguous()
    layer.rearrange_mixed_qkv = MethodType(gdn.QwenGatedDeltaNetAttention.rearrange_mixed_qkv, layer)

    def run(tokens, gating_a, gating_b, conv, recurrent, with_checkpoint=False):
        count = len(tokens)
        starts_cpu = torch.tensor([0, count], dtype=torch.int32)
        starts = starts_cpu.cuda()
        nums, batch_ptr, chunks = compute_causal_conv1d_metadata(starts_cpu, device=torch.device("cuda"))
        metadata = GDNAttentionMetadata(
            num_prefills=1, num_prefill_tokens=count, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=count,
            has_initial_state=torch.tensor([True], device="cuda"),
            non_spec_query_start_loc=starts,
            non_spec_state_indices_tensor=torch.tensor([1], device="cuda", dtype=torch.int32),
            prefill_query_start_loc=starts,
            prefill_state_indices=torch.tensor([1], device="cuda", dtype=torch.int32),
            prefill_has_initial_state=torch.tensor([True], device="cuda"),
            nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=chunks,
        )
        if with_checkpoint:
            setattr(metadata, checkpoints.CHECKPOINT_ATTRIBUTE, checkpoints.PrefillCheckpoint(
                torch.tensor([offset], device="cuda", dtype=torch.int32),
                torch.tensor([2], device="cuda", dtype=torch.int32), [offset],
            ))
        storage = conv if gdn.is_conv_state_dim_first() else conv.transpose(1, 2)
        layer.kv_cache = (storage, recurrent)
        monkeypatch.setattr(gdn, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"gdn": metadata}))
        output = torch.empty(count, heads, dimension, device="cuda", dtype=torch.bfloat16)
        gdn.QwenGatedDeltaNetAttention._forward_core(layer, tokens.clone(), gating_b, gating_a, output)
        return output

    baseline_conv, baseline_state = initial_conv.clone(), initial_recurrent.clone()
    baseline = run(raw, gating_a, gating_b, baseline_conv, baseline_state)
    conv, state = initial_conv.clone(), initial_recurrent.clone()
    actual = run(raw, gating_a, gating_b, conv, state, True)
    prefix_conv, prefix_state = initial_conv.clone(), initial_recurrent.clone()
    run(raw[:offset], gating_a[:offset], gating_b[:offset], prefix_conv, prefix_state)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(conv[1], baseline_conv[1], atol=0, rtol=0)
    torch.testing.assert_close(state[1], baseline_state[1], atol=0, rtol=0)
    torch.testing.assert_close(conv[2], prefix_conv[1], atol=0, rtol=0)
    torch.testing.assert_close(state[2], prefix_state[1], atol=0, rtol=0)
    conv[1], state[1] = conv[2].clone(), state[2].clone()
    continued = run(raw[offset:], gating_a[offset:], gating_b[offset:], conv, state)
    torch.testing.assert_close(continued, baseline[offset:], atol=0, rtol=0)
    torch.testing.assert_close(state[1], baseline_state[1], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("fused", [False, True])
def test_real_ple_dispatch_exports_history(monkeypatch, fused):
    import subprocess
    import sys

    script = (
        "import runpy, sys, pytest; "
        "namespace = runpy.run_path(sys.argv[1]); "
        "namespace['_check_real_ple_dispatch_history'](pytest.MonkeyPatch(), sys.argv[2] == 'True')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, __file__, str(fused)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_real_ple_dispatch_history(monkeypatch, fused):
    import importlib
    import inspect

    root = "vllm_hcu" if fused else "vllm"
    module = importlib.import_module(f"{root}.models.qwen4_exp.amd.ple_layer")
    _restore_layer_patch(monkeypatch, module, "Qwen4ExpPLELayer")
    checkpoints.patch_ple(module)
    monkeypatch.setenv("VLLM_HCU_MAMBA_PREFILL_CHECKPOINT", "1")
    length, offset, channels, dilation, width = 300, 128, 8, 2, 3
    window = (width - 1) * dilation
    layer = object.__new__(module.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.short_conv_dilation = dilation
    layer.conv_state_len = window
    layer.activation = "silu"
    inputs = torch.randn(length, channels, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(4, channels, window, device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=1, num_decodes=0, num_decode_tokens=0, num_prefill_tokens=length,
        num_actual_tokens=length, spec_sequence_masks=None, non_spec_token_indx=None,
        state_indices_tensor=torch.tensor([1], device="cuda", dtype=torch.int32),
        query_start_loc_p=starts, non_spec_query_start_loc=starts,
        has_initial_states_p=torch.tensor([False], device="cuda"),
        max_prefill_query_len=length,
    )
    setattr(metadata, checkpoints.CHECKPOINT_ATTRIBUTE, checkpoints.PrefillCheckpoint(
        torch.tensor([offset], device="cuda", dtype=torch.int32),
        torch.tensor([2], device="cuda", dtype=torch.int32), [offset],
    ))
    arguments = dict(inputs=inputs, metadata=metadata, conv_state=state, conv_weights=weights)
    signature = inspect.signature(layer._short_conv_dilated_dispatch)
    if "residual" in signature.parameters:
        arguments.update(residual=torch.zeros_like(inputs), outer_residual=torch.zeros_like(inputs))
    layer._short_conv_dilated_dispatch(**arguments)
    torch.testing.assert_close(state[2], inputs[offset - window:offset].T, atol=0, rtol=0)
    torch.testing.assert_close(state[1], inputs[-window:].T, atol=0, rtol=0)
    assert (state[[0, 3]] == 0).all()
