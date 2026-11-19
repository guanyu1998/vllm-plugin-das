"""Enable GDN internal checkpoints without changing vLLM source files."""

from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.prefill_checkpoint"
TARGETS = (
    f"{TARGET_MODULE}.QwenGatedDeltaNetAttention.get_kv_cache_spec",
    f"{TARGET_MODULE}.QwenGatedDeltaNetAttention._forward_core",
)


def apply_to_module(module):
    from vllm_hcu.model_executor.layers.mamba_prefill_checkpoint import patch_gdn

    module = load_exact_module(TARGET_MODULE, module)
    require_class(module, "QwenGatedDeltaNetAttention", TARGETS[0])
    return patch_gdn(module)


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
