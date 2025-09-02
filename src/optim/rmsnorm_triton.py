import os
import warnings
import torch
import torch.nn as nn


def _has_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:
        return False


if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_fwd(X, W, Y, N, eps, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * N + tl.arange(0, BLOCK)
        sumsq = tl.zeros([BLOCK], dtype=tl.float32)
        # Accumulate sum of squares across feature dim (N)
        for n in range(0, tl.cdiv(N, BLOCK)):
            idx = n * BLOCK + tl.arange(0, BLOCK)
            mask = idx < N
            x = tl.load(X + pid * N + idx, mask=mask, other=0.0)
            x = x.to(tl.float32)
            sumsq += x * x
        mean = tl.sum(sumsq, axis=0) / N
        inv = 1.0 / tl.sqrt(mean + eps)

        # Write normalized result with weight
        for n in range(0, tl.cdiv(N, BLOCK)):
            idx = n * BLOCK + tl.arange(0, BLOCK)
            mask = idx < N
            x = tl.load(X + pid * N + idx, mask=mask, other=0.0)
            w = tl.load(W + idx, mask=mask, other=1.0)
            y = (x * inv) * w
            tl.store(Y + pid * N + idx, y, mask=mask)


class TritonRMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape, "Last dim mismatch for RMSNorm"
        if x.numel() == 0:
            return x
        y = torch.empty_like(x)
        # Flatten batch dims, keep last dim as feature
        B = x.numel() // self.normalized_shape
        X = x.contiguous().view(B, self.normalized_shape)
        Y = y.view(B, self.normalized_shape)
        BLOCK = 128
        grid = (B,)
        _rmsnorm_fwd[grid](
            X, self.weight, Y, self.normalized_shape, self.eps, BLOCK=BLOCK
        )
        return y


def _replace_llama_rmsnorm(model: nn.Module, eps: float = 1e-5) -> int:
    """
    Replace LlamaRMSNorm modules with TritonRMSNorm. Returns number of replacements.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
    except Exception:
        return 0

    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LlamaRMSNorm):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent = model.get_submodule(parent_name) if parent_name else model
            child_name = name.split('.')[-1]
            triton_norm = TritonRMSNorm(module.weight.numel(), eps=module.variance_epsilon, dtype=module.weight.dtype)
            with torch.no_grad():
                triton_norm.weight.copy_(module.weight)
            setattr(parent, child_name, triton_norm)
            count += 1
    return count


def try_patch_rmsnorm(model: nn.Module) -> None:
    """
    If AZR_TRITON_RMSNORM=1 and Triton is available, replace Llama RMSNorm with TritonRMSNorm.
    """
    if os.environ.get("AZR_TRITON_RMSNORM", "0") != "1":
        return
    if not _has_triton():
        warnings.warn("AZR_TRITON_RMSNORM=1 but Triton is not available; skipping.")
        return
    try:
        n = _replace_llama_rmsnorm(model)
        if n == 0:
            warnings.warn("No LlamaRMSNorm modules found to replace.")
    except Exception as e:
        warnings.warn(f"Failed to replace RMSNorm with Triton version: {e}")

