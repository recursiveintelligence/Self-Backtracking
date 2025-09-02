import os
import warnings
import torch


def enable_torch_backend_flags():
    """
    Enable high‑performance CUDA paths where safe:
    - TF32 matmul for FP32 code paths (harmless when bf16 in use).
    - Prefer Flash/MemEff SDPA kernels over math fallback.
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")  # allow TF32 on Ampere/Hopper
    except Exception as e:
        warnings.warn(f"TF32 toggles not applied: {e}")

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        os.environ.setdefault("PYTORCH_SDP_KERNEL", "flash")
    except Exception as e:
        warnings.warn(f"SDPA toggles not applied: {e}")

    # Optional allocator tuning to reduce fragmentation on large models
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def prefer_sdpa_in_transformers(model):
    """
    Hint Transformers to use PyTorch SDPA path for attention when available.
    No-op if the model/config does not expose this knob.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "_attn_implementation"):
        try:
            # Transformers >= 4.31
            cfg._attn_implementation = "sdpa"
        except Exception:
            pass
    # Some models expose property instead of field
    if hasattr(model, "_attn_implementation"):
        try:
            model._attn_implementation = "sdpa"
        except Exception:
            pass


def maybe_compile(model, mode: str = "reduce-overhead"):
    """
    Optionally compile the model with TorchInductor.
    Controlled by env AZR_COMPILE=1 to avoid surprises.
    """
    use_compile = os.environ.get("AZR_COMPILE", "0") == "1"
    if not use_compile:
        return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False, dynamic=True)
        return compiled
    except Exception as e:
        warnings.warn(f"torch.compile failed; continuing without compile. Error: {e}")
        return model


def apply_performance_patches(model):
    """
    Apply a set of non-invasive performance improvements:
    - Enable CUDA backend flags.
    - Prefer SDPA attention in Transformers.
    - Optionally torch.compile when AZR_COMPILE=1.
    Returns (possibly wrapped) model.
    """
    enable_torch_backend_flags()
    prefer_sdpa_in_transformers(model)
    # Optionally replace RMSNorm with Triton implementation
    try:
        from .rmsnorm_triton import try_patch_rmsnorm
        try_patch_rmsnorm(model)
    except Exception:
        pass
    # Optional fused QKV (single GEMM for q/k/v) for LLaMA attention
    try:
        from .fused_qkv import try_patch_llama_fused_qkv
        try_patch_llama_fused_qkv(model)
    except Exception:
        pass
    # Optional RoPE fast path for LLaMA (Triton kernel)
    try:
        from .rope_triton import try_patch_rope
        try_patch_rope(model)
    except Exception:
        pass
    # Hint non-reentrant checkpointing when supported (lower overhead in PyTorch 2.3+)
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    model = maybe_compile(model)
    return model
