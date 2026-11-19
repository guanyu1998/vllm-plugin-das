"""Export PLE history at an internal prefill cache boundary."""

from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ple_layer"
PATCH_ID = "worker.core_fix.qwen4_exp.ple.prefill_checkpoint"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpPLELayer.get_kv_cache_spec",
    f"{TARGET_MODULE}.Qwen4ExpPLELayer._short_conv_dilated_dispatch",
)


def apply_to_module(module):
    from vllm_hcu.model_executor.layers.mamba_prefill_checkpoint import patch_ple

    if module.__name__ != "vllm_hcu.models.qwen4_exp.amd.ple_layer":
        module = load_exact_module(TARGET_MODULE, module)
    require_class(module, "Qwen4ExpPLELayer", TARGETS[0])
    return patch_ple(module)


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
