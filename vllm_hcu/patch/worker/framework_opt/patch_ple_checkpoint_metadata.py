"""Plan PLE checkpoint offsets using its own cache-group block table."""

from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.v1.attention.backends.short_conv_attn"
PATCH_ID = "worker.framework_opt.qwen4_exp.ple.checkpoint_metadata"
TARGETS = (f"{TARGET_MODULE}.PleShortConvAttentionMetadataBuilder.build",)


def apply_to_module(module):
    from vllm_hcu.model_executor.layers.mamba_prefill_checkpoint import (
        patch_metadata_builder,
        patch_metadata_type,
    )

    module = load_exact_module(TARGET_MODULE, module)
    builder = require_class(module, "PleShortConvAttentionMetadataBuilder", TARGETS[0])
    patch_metadata_type(module, builder, "PleShortConvAttentionMetadata")
    return patch_metadata_builder(builder)


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
