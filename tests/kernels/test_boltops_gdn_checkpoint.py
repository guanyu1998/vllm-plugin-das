# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate the standalone recurrence against the unmodified BoltOps recurrence.

BoltOps is only a numeric test oracle; it is not a kernel runtime dependency.
"""

import pytest
import torch

checkpoint_kernel = pytest.importorskip("boltops_gdn_kernel").chunk_gated_delta_rule_fwd_h

baseline = pytest.importorskip("boltops.fla.gdn.triton.chunk_delta_h")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")


def make_inputs(lengths=(193, 129, 71), key_dim=128, value_dim=64,
                heads=3, dtype=torch.bfloat16, transpose=True, batch=1):
    torch.manual_seed(20261119)
    total = sum(lengths)
    key = torch.randn(batch, total, 1, key_dim, device="cuda", dtype=dtype) * 0.03
    weights = torch.randn(batch, total, heads, key_dim, device="cuda", dtype=dtype) * 0.03
    values = torch.randn(batch, total, heads, value_dim, device="cuda", dtype=dtype) * 0.03
    gate = torch.empty(batch, total, heads, device="cuda")
    start = 0
    for length in lengths:
        for offset in range(0, length, 64):
            end = min(offset + 64, length)
            gate[:, start + offset:start + end] = (
                -torch.rand(batch, end - offset, heads, device="cuda") * 0.01
            ).cumsum(1)
        start += length
    shape = (heads, value_dim, key_dim) if transpose else (heads, key_dim, value_dim)
    initial = torch.randn(len(lengths) if batch == 1 else batch, *shape, device="cuda") * 0.01
    return dict(k=key, w=weights, u=values, g=gate, initial_state=initial,
                output_final_state=True, transpose_state_layout=transpose,
                cu_seqlens=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                                        device="cuda", dtype=torch.int32) if batch == 1 else None)


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("key_dim,value_dim", [(64, 80), (128, 128), (256, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_varlen_matches_original_and_stopped_prefix(transpose, key_dim, value_dim, dtype):
    inputs = make_inputs(key_dim=key_dim, value_dim=value_dim, transpose=transpose, dtype=dtype)
    if key_dim == 256:
        # The installed pre-extension ABI requires a complete override dict.
        inputs["kernel_cfg"] = {"BV": 32, "num_warps": 8, "num_stages": 1}
    offsets = torch.tensor([128, 64, 0], device="cuda", dtype=torch.int64)
    destination = torch.full_like(inputs["initial_state"], 7)
    actual = checkpoint_kernel(**inputs, checkpoint_state=destination, checkpoint_offsets=offsets)
    expected = baseline.chunk_gated_delta_rule_fwd_h(**inputs)
    for result, reference in zip(actual, expected):
        exact(result, reference)
    for row, start, offset in [(0, 0, 128), (1, 193, 64)]:
        prefix = {name: tensor[:, start:start + offset].contiguous()
                  for name, tensor in inputs.items() if name in ("k", "w", "u", "g")}
        stopped = baseline.chunk_gated_delta_rule_fwd_h(
            **prefix, initial_state=inputs["initial_state"][row:row + 1],
            transpose_state_layout=transpose, output_final_state=True,
            kernel_cfg=inputs.get("kernel_cfg"),
        )[2]
        exact(destination[row], stopped[0])
    assert (destination[2] == 7).all()


@pytest.mark.parametrize("offset", [-64, 0, 63, 193, 256])
def test_invalid_offsets_leave_destination_untouched(offset):
    inputs = make_inputs(lengths=(193,))
    destination = torch.full_like(inputs["initial_state"], 9)
    checkpoint_kernel(**inputs, checkpoint_state=destination,
                      checkpoint_offsets=torch.tensor([offset], device="cuda", dtype=torch.int32))
    assert (destination == 9).all()


@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("null_index", [-1, 0])
def test_direct_strided_cache_and_invalid_rows(indices_dtype, null_index):
    inputs = make_inputs()
    shape = inputs["initial_state"].shape[1:]
    row_size = inputs["initial_state"][0].numel()
    storage = torch.full((6 * (row_size + 37),), 11., device="cuda")
    destination = storage.as_strided((6, *shape), (row_size + 37, *inputs["initial_state"].stride()[1:]))
    offsets = torch.tensor([128, 64, 64], device="cuda", dtype=indices_dtype)
    indices = torch.tensor([4, null_index, 9], device="cuda", dtype=indices_dtype)
    checkpoint_kernel(**inputs, checkpoint_state=destination, checkpoint_offsets=offsets,
                      checkpoint_state_indices=indices, null_state_index=null_index)
    assert (destination[[0, 1, 2, 3, 5]] == 11).all()
    assert (destination[4] != 11).any()
    assert (storage[row_size:row_size + 37] == 11).all()


def test_uniform_batch_graph_capture_and_resume():
    inputs = make_inputs(lengths=(256,), batch=2)
    destination = torch.empty_like(inputs["initial_state"])
    offsets = torch.tensor([128, 64], device="cuda", dtype=torch.int32)
    checkpoint_kernel(**inputs, checkpoint_state=destination, checkpoint_offsets=offsets)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = checkpoint_kernel(**inputs, checkpoint_state=destination, checkpoint_offsets=offsets)
    graph.replay()
    expected = baseline.chunk_gated_delta_rule_fwd_h(**inputs)
    for result, reference in zip(actual, expected):
        exact(result, reference)
    for row, offset in enumerate((128, 64)):
        tail = {name: tensor[row:row + 1, offset:].contiguous()
                for name, tensor in inputs.items() if name in ("k", "w", "u", "g")}
        continued = baseline.chunk_gated_delta_rule_fwd_h(
            **tail, initial_state=destination[row:row + 1], output_final_state=True,
        )
        exact(continued[2][0], expected[2][row])


def test_optional_final_and_inplace_do_not_change_checkpoint():
    inputs = make_inputs(lengths=(193,))
    offsets = torch.tensor([128], device="cuda", dtype=torch.int32)
    expected = torch.empty_like(inputs["initial_state"])
    checkpoint_kernel(**inputs, checkpoint_state=expected, checkpoint_offsets=offsets)
    for inplace, final, save in [(False, False, False), (True, True, True)]:
        destination = torch.empty_like(expected)
        arguments = dict(inputs, initial_state=inputs["initial_state"].clone(),
                         output_final_state=final, inplace_final_state=inplace, save_new_value=save)
        result = checkpoint_kernel(**arguments, checkpoint_state=destination, checkpoint_offsets=offsets)
        exact(destination, expected)
        assert (result[1] is not None) == save
        assert (result[2] is not None) == final


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_dense_initial_state_row_zero_is_live(with_checkpoint):
    inputs = make_inputs(lengths=(193,))
    arguments = {}
    if with_checkpoint:
        arguments = dict(checkpoint_state=torch.empty_like(inputs["initial_state"]),
                         checkpoint_offsets=torch.tensor([128], device="cuda", dtype=torch.int32))
    actual = checkpoint_kernel(**inputs, null_state_index=0, **arguments)
    expected = baseline.chunk_gated_delta_rule_fwd_h(**inputs, null_state_index=-1)
    for result, reference in zip(actual, expected):
        exact(result, reference)


def test_aligned_terminal_offset_is_not_exported():
    inputs = make_inputs(lengths=(256,))
    destination = torch.full_like(inputs["initial_state"], 17)
    checkpoint_kernel(**inputs, checkpoint_state=destination,
                      checkpoint_offsets=torch.tensor([256], device="cuda", dtype=torch.int32))
    assert (destination == 17).all()


def test_empty_sequence_leaves_checkpoint_untouched():
    inputs = make_inputs(lengths=(0, 193))
    destination = torch.full_like(inputs["initial_state"], 19)
    actual = checkpoint_kernel(
        **inputs, checkpoint_state=destination,
        checkpoint_offsets=torch.tensor([64, 128], device="cuda", dtype=torch.int32))
    expected = baseline.chunk_gated_delta_rule_fwd_h(**inputs)
    for result, reference in zip(actual, expected):
        exact(result, reference)
    assert (destination[0] == 19).all()
    assert (destination[1] != 19).any()


@pytest.mark.parametrize("gating", ["none", "gk", "exp2"])
def test_gating_and_large_cache_stride(gating):
    inputs = make_inputs(lengths=(193,), heads=1, value_dim=32)
    if gating == "none":
        inputs["g"] = None
    elif gating == "gk":
        inputs["gk"] = torch.zeros(1, 193, 1, 128, device="cuda")
    else:
        inputs["use_exp2"] = True
    stride = 2**30 + 128
    storage = torch.empty(2 * stride + 32 * 128, device="cuda")
    destination = storage.as_strided((3, 1, 32, 128), (stride, 4096, 128, 1))
    destination[0].fill_(13)
    offsets = torch.tensor([128], device="cuda", dtype=torch.int32)
    indices = torch.tensor([2], device="cuda", dtype=torch.int64)
    checkpoint_kernel(**inputs, checkpoint_state=destination, checkpoint_offsets=offsets,
                      checkpoint_state_indices=indices)
    prefix = dict(inputs)
    prefix["cu_seqlens"] = None
    for name in ("k", "w", "u", "g", "gk"):
        if prefix.get(name) is not None:
            prefix[name] = prefix[name][:, :128].contiguous()
    exact(destination[2], baseline.chunk_gated_delta_rule_fwd_h(**prefix)[2][0])
    assert (destination[0] == 13).all()


@pytest.mark.parametrize("invalid", ["missing_state", "missing_offsets", "dtype", "offset_dtype",
                                    "offset_shape", "state_shape", "state_stride", "indices"])
def test_validation(invalid):
    inputs = make_inputs(lengths=(193,))
    state = torch.empty_like(inputs["initial_state"])
    arguments = dict(checkpoint_state=state,
                     checkpoint_offsets=torch.tensor([128], device="cuda", dtype=torch.int32))
    if invalid == "missing_state":
        arguments.pop("checkpoint_state")
    elif invalid == "missing_offsets":
        arguments.pop("checkpoint_offsets")
    elif invalid == "dtype":
        arguments["checkpoint_state"] = state.bfloat16()
    elif invalid == "offset_dtype":
        arguments["checkpoint_offsets"] = arguments["checkpoint_offsets"].float()
    elif invalid == "offset_shape":
        arguments["checkpoint_offsets"] = torch.ones(2, device="cuda", dtype=torch.int32)
    elif invalid == "state_shape":
        arguments["checkpoint_state"] = state[:, :, :, :-1]
    elif invalid == "state_stride":
        arguments["checkpoint_state"] = state.transpose(2, 3)
    else:
        arguments["checkpoint_state_indices"] = torch.ones(2, device="cuda", dtype=torch.int32)
    with pytest.raises((ValueError, TypeError)):
        checkpoint_kernel(**inputs, **arguments)


@pytest.mark.parametrize("heads", [24, 48])
@pytest.mark.parametrize("length", [256, 2048, 8192, 35263])
def test_tuned_production_shapes_match_original_and_prefix(heads, length):
    from boltops_gdn_kernel.chunk_delta_h import (
        _get_chunk_delta_h_config,
    )

    inputs = make_inputs(lengths=(length,), heads=heads, value_dim=128)
    # The tuned entries apply to Hv=3*Hk, including multiple key heads.
    inputs["k"] = torch.randn(1, length, heads // 3, 128, device="cuda",
                              dtype=torch.bfloat16) * 0.03
    offset = ((length - 1) // 64) * 64
    offsets = torch.tensor([offset], device="cuda", dtype=torch.int32)
    indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    state = torch.full((4, heads, 128, 128), 7.0, device="cuda")
    actual = checkpoint_kernel(**inputs, checkpoint_state=state,
                               checkpoint_offsets=offsets,
                               checkpoint_state_indices=indices)
    expected = baseline.chunk_gated_delta_rule_fwd_h(**inputs)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-2, atol=2e-3)
    prefix = dict(inputs, cu_seqlens=None)
    for name in ("k", "w", "u", "g"):
        prefix[name] = prefix[name][:, :offset].contiguous()
    stopped = baseline.chunk_gated_delta_rule_fwd_h(**prefix)[2]
    torch.testing.assert_close(state[2], stopped[0], rtol=2e-2, atol=2e-3)
    assert (state[[0, 1, 3]] == 7).all()
    # Compare the automatic tuned dispatch with an explicit fixed launch.
    config = _get_chunk_delta_h_config(128, 128, 64, heads, True, True)
    fixed = checkpoint_kernel(**inputs, kernel_cfg=config)
    for value, reference in zip(actual, fixed):
        exact(value, reference)
    tail = dict(inputs, initial_state=state[2:3], cu_seqlens=None, kernel_cfg=config)
    for name in ("k", "w", "u", "g"):
        tail[name] = tail[name][:, offset:].contiguous()
    continued = checkpoint_kernel(**tail)
    exact(continued[2], actual[2])
