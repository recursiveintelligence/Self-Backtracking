import os
import warnings
import torch
import torch.nn as nn
from torch.autograd import Function


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
        # Compute row base pointer
        row_x = X + pid * N
        row_y = Y + pid * N

        # 1) Reduce sum of squares across the feature dimension
        accum = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, tl.cdiv(N, BLOCK)):
            idx = off * BLOCK + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(row_x + idx, mask=m, other=0.0)
            x = x.to(tl.float32)
            accum += x * x
        mean = tl.sum(accum, axis=0) / N
        inv = 1.0 / tl.sqrt(mean + eps)

        # 2) Normalize and apply weight
        for off in range(0, tl.cdiv(N, BLOCK)):
            idx = off * BLOCK + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(row_x + idx, mask=m, other=0.0)
            w = tl.load(W + idx, mask=m, other=1.0)
            y = (x.to(tl.float32) * inv) * w.to(tl.float32)
            if tl.typeof(Y) == tl.pointer_type(tl.float16):
                y = y.to(tl.float16)
            elif tl.typeof(Y) == tl.pointer_type(tl.bfloat16):
                y = y.to(tl.bfloat16)
            tl.store(row_y + idx, y, mask=m)

    @triton.jit
    def _rmsnorm_bwd(X, W, DY, DX, DW, N, eps, BLOCK: tl.constexpr):
        """
        Backward for RMSNorm (per-row program):
        Given y = w * x * r, r = 1/sqrt(mean(x^2)+eps)
        Let g = dy * w.
        dx = r * g - (r^3 / N) * x * sum(g * x)
        dW = sum_over_rows(dy * x * r)
        DW is float32 accumulation buffer (global reduction via atomics).
        """
        pid = tl.program_id(0)
        row_x = X + pid * N
        row_dy = DY + pid * N
        row_dx = DX + pid * N

        # 1) Compute r and dot = sum(g * x)
        # accumulate sumsq to compute r
        accum = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, tl.cdiv(N, BLOCK)):
            idx = off * BLOCK + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(row_x + idx, mask=m, other=0.0).to(tl.float32)
            accum += x * x
        mean = tl.sum(accum, axis=0) / N
        r = 1.0 / tl.sqrt(mean + eps)

        # compute dot = sum(g * x) with g = dy * w
        dot_acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, tl.cdiv(N, BLOCK)):
            idx = off * BLOCK + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(row_x + idx, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(row_dy + idx, mask=m, other=0.0).to(tl.float32)
            w = tl.load(W + idx, mask=m, other=1.0).to(tl.float32)
            g = dy * w
            dot_acc += g * x
        dot = tl.sum(dot_acc, axis=0)

        # 2) Write dx and accumulate dW
        scale = r
        coeff = (r * r * r) / N  # r^3 / N
        for off in range(0, tl.cdiv(N, BLOCK)):
            idx = off * BLOCK + tl.arange(0, BLOCK)
            m = idx < N
            x = tl.load(row_x + idx, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(row_dy + idx, mask=m, other=0.0).to(tl.float32)
            w = tl.load(W + idx, mask=m, other=1.0).to(tl.float32)
            g = dy * w
            dx = scale * g - coeff * x * dot
            # Store dx in the same dtype as input
            if tl.typeof(DX) == tl.pointer_type(tl.float16):
                dx_cast = dx.to(tl.float16)
            elif tl.typeof(DX) == tl.pointer_type(tl.bfloat16):
                dx_cast = dx.to(tl.bfloat16)
            else:
                dx_cast = dx
            tl.store(row_dx + idx, dx_cast, mask=m)
            # dW contribution: dy * x * r (accumulate in fp32)
            dwi = dy * x * r
            tl.atomic_add(DW + idx, dwi, mask=m)


class _TritonRMSNormFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        assert x.is_cuda and weight.is_cuda, "Triton RMSNorm requires CUDA tensors"
        N = x.shape[-1]
        B = x.numel() // N
        x_2d = x.contiguous().view(B, N)
        y = torch.empty_like(x_2d)
        BLOCK = 128
        grid = (B,)
        _rmsnorm_fwd[grid](
            x_2d, weight, y, N, eps, BLOCK=BLOCK
        )
        ctx.save_for_backward(x_2d, weight)
        ctx.N = N
        ctx.eps = eps
        return y.view_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w = ctx.saved_tensors
        N, eps = ctx.N, ctx.eps
        B = x.shape[0]

        dy_2d = dy.contiguous().view(B, N)
        dx = torch.empty_like(dy_2d)
        # Accumulate dW in fp32 for numerical stability
        dW_accum = torch.zeros_like(w, dtype=torch.float32)

        BLOCK = 128
        grid = (B,)
        _rmsnorm_bwd[grid](
            x, w, dy_2d, dx, dW_accum, N, eps, BLOCK=BLOCK
        )
        # Match parameter dtype
        dW = dW_accum.to(w.dtype)
        return dx.view_as(dy), dW, None


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
        if not x.is_cuda:
            # CPU fallback uses PyTorch ops (keeps autograd)
            var = (x.to(torch.float32) ** 2).mean(dim=-1, keepdim=True)
            r = torch.rsqrt(var + self.eps)
            y = x * r * self.weight
            return y
        return _TritonRMSNormFn.apply(x, self.weight, self.eps)


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
            # Match device/dtype to replaced module's weight
            triton_norm = triton_norm.to(device=module.weight.device, dtype=module.weight.dtype)
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
