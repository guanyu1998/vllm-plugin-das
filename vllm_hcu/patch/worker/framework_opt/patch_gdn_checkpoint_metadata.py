"""Plan GDN checkpoint offsets and destination cache slots."""

from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.v1.attention.backends.gdn_attn"
PATCH_ID = "worker.framework_opt.mamba.gdn.checkpoint_metadata"
TARGETS = (f"{TARGET_MODULE}.GDNAttentionMetadataBuilder.build",)


def apply_to_module(module):
    from vllm_hcu.model_executor.layers.mamba_prefill_checkpoint import (
        patch_metadata_builder,
        patch_metadata_type,
    )

    module = load_exact_module(TARGET_MODULE, module)
    builder = require_class(module, "GDNAttentionMetadataBuilder", TARGETS[0])
    patch_metadata_type(module, builder, "GDNAttentionMetadata")
    return patch_metadata_builder(builder)


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
