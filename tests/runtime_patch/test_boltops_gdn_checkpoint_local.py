# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Check plugin checkpoint routing without a modified BoltOps installation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_external_kernel_import_without_hcu_or_boltops():
    package = pytest.importorskip("boltops_gdn_kernel")
    root = Path(package.__file__).resolve().parent.parent
    script = r'''
import importlib.abc
import sys
class StandaloneOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"vllm_hcu", "vllm", "boltops"}:
            raise AssertionError("Unexpected kernel dependency: " + fullname)
sys.meta_path.insert(0, StandaloneOnly())
import torch
from boltops_gdn_kernel import chunk_gated_delta_rule_fwd_h
from boltops_gdn_kernel.chunk_delta_h import _get_chunk_delta_h_config
assert chunk_gated_delta_rule_fwd_h.__module__ == "boltops_gdn_kernel.chunk_delta_h"
assert _get_chunk_delta_h_config(128, 128, 64, 24, True, True)["BV"] == 64
assert _get_chunk_delta_h_config(128, 128, 64, 48, True, True)["BK"] == 128
assert not torch.cuda.is_initialized()
'''
    env = {**os.environ, "PYTHONPATH": str(root), "HIP_VISIBLE_DEVICES": "",
           "ROCR_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing", ["boltops_gdn_kernel", "triton"])
def test_optional_external_resolver_preserves_dependency_errors(monkeypatch, missing):
    from vllm_hcu.model_executor.layers import mamba_prefill_checkpoint as checkpoint

    def unavailable(name):
        assert name == "boltops_gdn_kernel"
        raise ModuleNotFoundError("unavailable", name=missing)

    checkpoint.boltops_checkpoint_kernel.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(checkpoint.importlib, "import_module", unavailable)
            if missing == "boltops_gdn_kernel":
                assert checkpoint.boltops_checkpoint_kernel() is None
            else:
                with pytest.raises(ModuleNotFoundError) as failure:
                    checkpoint.boltops_checkpoint_kernel()
                assert failure.value.name == "triton"
    finally:
        checkpoint.boltops_checkpoint_kernel.cache_clear()


def test_plugin_import_resolves_external_kernel():
    package = pytest.importorskip("boltops_gdn_kernel")
    from vllm_hcu.model_executor.layers import mamba_prefill_checkpoint as checkpoint
    from vllm_hcu.model_executor.layers.boltops_gdn import chunk_gated_delta_rule_fwd_h

    assert checkpoint.boltops_checkpoint_kernel() is package.chunk_gated_delta_rule_fwd_h
    assert chunk_gated_delta_rule_fwd_h is package.chunk_gated_delta_rule_fwd_h


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU")
@pytest.mark.parametrize("cache_dtype", ["float32", "bfloat16"])
def test_local_checkpoint_and_gdn_core_without_boltops(cache_dtype):
    pytest.importorskip("boltops_gdn_kernel")
    # A fresh process proves neither importing nor executing the kernel needs
    # BoltOps, including the real GDN checkpoint and continuation path.
    script = r'''
import importlib.abc
import runpy
import sys

class NoBoltOps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "boltops" or fullname.startswith("boltops."):
            raise ModuleNotFoundError("BoltOps deliberately unavailable", name=fullname)

sys.meta_path.insert(0, NoBoltOps())
import pytest
import torch
from boltops_gdn_kernel import chunk_gated_delta_rule_fwd_h
from vllm_hcu.model_executor.layers import mamba_prefill_checkpoint as checkpoint
from vllm_hcu.patch.worker.op_opt import _gdn_backend

assert checkpoint.boltops_checkpoint_kernel() is chunk_gated_delta_rule_fwd_h
from vllm_hcu.model_executor.layers.boltops_gdn import chunk_gated_delta_rule_fwd_h as compatibility_import
assert compatibility_import is chunk_gated_delta_rule_fwd_h
assert not any(name == "boltops" or name.startswith("boltops.") for name in sys.modules)
torch.manual_seed(19)
length, offset, heads, width = 257, 128, 3, 128
q = torch.randn(1, length, 1, width, device="cuda", dtype=torch.bfloat16)
inputs = dict(q=q, k=torch.randn_like(q),
              v=torch.randn(1, length, heads, width, device="cuda", dtype=torch.bfloat16),
              g=torch.nn.functional.logsigmoid(torch.randn(1, length, heads, device="cuda")),
              beta=torch.rand(1, length, heads, device="cuda"))
initial = torch.randn(1, heads, width, width, device="cuda") * .01
dtype = getattr(torch, sys.argv[1])
state = torch.full((4, heads, width, width), 7., device="cuda", dtype=dtype)
starts = torch.tensor([0, length], device="cuda", dtype=torch.int32)
plan = checkpoint.PrefillCheckpoint(
    torch.tensor([offset], device="cuda", dtype=torch.int32),
    torch.tensor([2], device="cuda", dtype=torch.int32), [offset])
if dtype == torch.float32:
    class NoCopy:
        def __getitem__(self, grid):
            raise AssertionError("FP32 checkpoint must be written directly")
    checkpoint._store_recurrent_checkpoint = NoCopy()
actual, final = checkpoint.checkpoint_chunk(
    **inputs, initial_state=initial, output_final_state=True,
    use_qk_l2norm_in_kernel=True, cu_seqlens=starts,
    checkpoint=plan, state=state, checkpoint_backend="hcu-boltops")
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
reference, reference_final = chunk_gated_delta_rule(
    **inputs, initial_state=initial, output_final_state=True,
    use_qk_l2norm_in_kernel=True, cu_seqlens=starts)
torch.testing.assert_close(actual, reference, rtol=0, atol=0)
torch.testing.assert_close(final, reference_final, rtol=0, atol=0)
prefix = {name: tensor[:, :offset].contiguous() for name, tensor in inputs.items()}
_, stopped = chunk_gated_delta_rule(
    **prefix, initial_state=initial, output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=torch.tensor([0, offset], device="cuda", dtype=torch.int32))
torch.testing.assert_close(state[2], stopped[0].to(dtype), rtol=0, atol=0)
assert (state[[0, 1, 3]] == 7).all()

# The provider decision in the patched real GDN core must select our local
# recurrence even though the external BoltOps resolver is unavailable.
_gdn_backend._prefill_requested = "hcu-boltops"
namespace = runpy.run_path(sys.argv[2])
for transposed in (False, True):
    with pytest.MonkeyPatch.context() as monkeypatch:
        namespace["test_real_gdn_core_exports_both_states_and_resumes"](
            monkeypatch, transposed)
assert not any(name == "boltops" or name.startswith("boltops.") for name in sys.modules)
'''
    test_path = Path(__file__).with_name("test_mamba_prefill_checkpoint.py")
    result = subprocess.run(
        [sys.executable, "-c", script, cache_dtype, str(test_path)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
