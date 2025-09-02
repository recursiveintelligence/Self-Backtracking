import os
import warnings
from typing import Tuple

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


_ROPE_LAST_INV_FREQ = None  # module-level capture from rotary_emb


def set_last_inv_freq(t: torch.Tensor):
    global _ROPE_LAST_INV_FREQ
    _ROPE_LAST_INV_FREQ = t.detach().to(torch.float32)


if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_fwd(X, COS, SIN, Y, N, RD, BLOCK: tl.constexpr):
        """
        Apply RoPE to a single 2D matrix [M, N] row-wise.
        - X: input [M, N]
        - COS, SIN: per-row cos/sin aligned [M, RD]
        - Y: output [M, N]
        - N: head_dim; RD: rotary_dim (<= N)
        Handles both even/odd with elementwise cos/sin.
        """
        pid = tl.program_id(0)
        row_x = X + pid * N
        row_y = Y + pid * N
        row_cos = COS + pid * RD
        row_sin = SIN + pid * RD

        # 1) RoPE on first RD dims in pairs
        # Iterate pairs (0::2, 1::2)
        num_pairs = RD // 2
        for p in range(0, tl.cdiv(num_pairs, BLOCK)):
            ip = p * BLOCK + tl.arange(0, BLOCK)
            m = ip < num_pairs
            idx_e = ip * 2
            idx_o = idx_e + 1

            x_e = tl.load(row_x + idx_e, mask=m, other=0.0)
            x_o = tl.load(row_x + idx_o, mask=m, other=0.0)
            c_e = tl.load(row_cos + idx_e, mask=m, other=1.0)
            c_o = tl.load(row_cos + idx_o, mask=m, other=1.0)
            s_e = tl.load(row_sin + idx_e, mask=m, other=0.0)
            s_o = tl.load(row_sin + idx_o, mask=m, other=0.0)

            # y_even = x_e * c_e - x_o * s_e
            # y_odd  = x_o * c_o + x_e * s_o
            y_e = x_e.to(tl.float32) * c_e.to(tl.float32) - x_o.to(tl.float32) * s_e.to(tl.float32)
            y_o = x_o.to(tl.float32) * c_o.to(tl.float32) + x_e.to(tl.float32) * s_o.to(tl.float32)

            # Cast to output dtype
            if tl.typeof(Y) == tl.pointer_type(tl.float16):
                y_e = y_e.to(tl.float16)
                y_o = y_o.to(tl.float16)
            elif tl.typeof(Y) == tl.pointer_type(tl.bfloat16):
                y_e = y_e.to(tl.bfloat16)
                y_o = y_o.to(tl.bfloat16)
            tl.store(row_y + idx_e, y_e, mask=m)
            tl.store(row_y + idx_o, y_o, mask=m)

        # 2) Pass-through remaining dims
        for off in range(RD, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            v = tl.load(row_x + idx, mask=m, other=0.0)
            tl.store(row_y + idx, v, mask=m)

    @triton.jit
    def _rope_bwd(DY, COS, SIN, DX, N, RD, BLOCK: tl.constexpr):
        """
        Backward w.r.t X only (cos/sin are constants):
        y_even = x_e * c_e - x_o * s_e
        y_odd  = x_o * c_o + x_e * s_o
        =>
        dx_e = dy_e * c_e + dy_o * s_o
        dx_o = -dy_e * s_e + dy_o * c_o
        """
        pid = tl.program_id(0)
        row_dy = DY + pid * N
        row_dx = DX + pid * N
        row_cos = COS + pid * RD
        row_sin = SIN + pid * RD

        # 1) RoPE grad for first RD dims
        num_pairs = RD // 2
        for p in range(0, tl.cdiv(num_pairs, BLOCK)):
            ip = p * BLOCK + tl.arange(0, BLOCK)
            m = ip < num_pairs
            idx_e = ip * 2
            idx_o = idx_e + 1

            dy_e = tl.load(row_dy + idx_e, mask=m, other=0.0).to(tl.float32)
            dy_o = tl.load(row_dy + idx_o, mask=m, other=0.0).to(tl.float32)
            c_e = tl.load(row_cos + idx_e, mask=m, other=1.0).to(tl.float32)
            c_o = tl.load(row_cos + idx_o, mask=m, other=1.0).to(tl.float32)
            s_e = tl.load(row_sin + idx_e, mask=m, other=0.0).to(tl.float32)
            s_o = tl.load(row_sin + idx_o, mask=m, other=0.0).to(tl.float32)

            dx_e = dy_e * c_e + dy_o * s_o
            dx_o = -dy_e * s_e + dy_o * c_o

            # Cast to output dtype
            if tl.typeof(DX) == tl.pointer_type(tl.float16):
                dx_e = dx_e.to(tl.float16)
                dx_o = dx_o.to(tl.float16)
            elif tl.typeof(DX) == tl.pointer_type(tl.bfloat16):
                dx_e = dx_e.to(tl.bfloat16)
                dx_o = dx_o.to(tl.bfloat16)
            tl.store(row_dx + idx_e, dx_e, mask=m)
            tl.store(row_dx + idx_o, dx_o, mask=m)

        # 2) Pass-through for remaining dims
        for off in range(RD, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            v = tl.load(row_dy + idx, mask=m, other=0.0)
            tl.store(row_dx + idx, v, mask=m)

    @triton.jit
    def _rope2_fwd(XQ, XK, COS, SIN, YQ, YK, N, RD, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_xq = XQ + pid * N
        row_xk = XK + pid * N
        row_yq = YQ + pid * N
        row_yk = YK + pid * N
        row_cos = COS + pid * RD
        row_sin = SIN + pid * RD

        num_pairs = RD // 2
        for p in range(0, tl.cdiv(num_pairs, BLOCK)):
            ip = p * BLOCK + tl.arange(0, BLOCK)
            m = ip < num_pairs
            idx_e = ip * 2
            idx_o = idx_e + 1

            q_e = tl.load(row_xq + idx_e, mask=m, other=0.0)
            q_o = tl.load(row_xq + idx_o, mask=m, other=0.0)
            k_e = tl.load(row_xk + idx_e, mask=m, other=0.0)
            k_o = tl.load(row_xk + idx_o, mask=m, other=0.0)
            c_e = tl.load(row_cos + idx_e, mask=m, other=1.0)
            c_o = tl.load(row_cos + idx_o, mask=m, other=1.0)
            s_e = tl.load(row_sin + idx_e, mask=m, other=0.0)
            s_o = tl.load(row_sin + idx_o, mask=m, other=0.0)

            yq_e = q_e.to(tl.float32) * c_e.to(tl.float32) - q_o.to(tl.float32) * s_e.to(tl.float32)
            yq_o = q_o.to(tl.float32) * c_o.to(tl.float32) + q_e.to(tl.float32) * s_o.to(tl.float32)
            yk_e = k_e.to(tl.float32) * c_e.to(tl.float32) - k_o.to(tl.float32) * s_e.to(tl.float32)
            yk_o = k_o.to(tl.float32) * c_o.to(tl.float32) + k_e.to(tl.float32) * s_o.to(tl.float32)

            if tl.typeof(YQ) == tl.pointer_type(tl.float16):
                yq_e = yq_e.to(tl.float16); yq_o = yq_o.to(tl.float16)
            elif tl.typeof(YQ) == tl.pointer_type(tl.bfloat16):
                yq_e = yq_e.to(tl.bfloat16); yq_o = yq_o.to(tl.bfloat16)
            if tl.typeof(YK) == tl.pointer_type(tl.float16):
                yk_e = yk_e.to(tl.float16); yk_o = yk_o.to(tl.float16)
            elif tl.typeof(YK) == tl.pointer_type(tl.bfloat16):
                yk_e = yk_e.to(tl.bfloat16); yk_o = yk_o.to(tl.bfloat16)
            tl.store(row_yq + idx_e, yq_e, mask=m)
            tl.store(row_yq + idx_o, yq_o, mask=m)
            tl.store(row_yk + idx_e, yk_e, mask=m)
            tl.store(row_yk + idx_o, yk_o, mask=m)

        for off in range(RD, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            vq = tl.load(row_xq + idx, mask=m, other=0.0)
            vk = tl.load(row_xk + idx, mask=m, other=0.0)
            tl.store(row_yq + idx, vq, mask=m)
            tl.store(row_yk + idx, vk, mask=m)

    @triton.jit
    def _rope2_bwd(DYQ, DYK, COS, SIN, DXQ, DXK, N, RD, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_dyq = DYQ + pid * N
        row_dyk = DYK + pid * N
        row_dxq = DXQ + pid * N
        row_dxk = DXK + pid * N
        row_cos = COS + pid * RD
        row_sin = SIN + pid * RD

        num_pairs = RD // 2
        for p in range(0, tl.cdiv(num_pairs, BLOCK)):
            ip = p * BLOCK + tl.arange(0, BLOCK)
            m = ip < num_pairs
            idx_e = ip * 2
            idx_o = idx_e + 1

            dyq_e = tl.load(row_dyq + idx_e, mask=m, other=0.0).to(tl.float32)
            dyq_o = tl.load(row_dyq + idx_o, mask=m, other=0.0).to(tl.float32)
            dyk_e = tl.load(row_dyk + idx_e, mask=m, other=0.0).to(tl.float32)
            dyk_o = tl.load(row_dyk + idx_o, mask=m, other=0.0).to(tl.float32)
            c_e = tl.load(row_cos + idx_e, mask=m, other=1.0).to(tl.float32)
            c_o = tl.load(row_cos + idx_o, mask=m, other=1.0).to(tl.float32)
            s_e = tl.load(row_sin + idx_e, mask=m, other=0.0).to(tl.float32)
            s_o = tl.load(row_sin + idx_o, mask=m, other=0.0).to(tl.float32)

            dxq_e = dyq_e * c_e + dyq_o * s_o
            dxq_o = -dyq_e * s_e + dyq_o * c_o
            dxk_e = dyk_e * c_e + dyk_o * s_o
            dxk_o = -dyk_e * s_e + dyk_o * c_o

            if tl.typeof(DXQ) == tl.pointer_type(tl.float16):
                dxq_e = dxq_e.to(tl.float16); dxq_o = dxq_o.to(tl.float16)
            elif tl.typeof(DXQ) == tl.pointer_type(tl.bfloat16):
                dxq_e = dxq_e.to(tl.bfloat16); dxq_o = dxq_o.to(tl.bfloat16)
            if tl.typeof(DXK) == tl.pointer_type(tl.float16):
                dxk_e = dxk_e.to(tl.float16); dxk_o = dxk_o.to(tl.float16)
            elif tl.typeof(DXK) == tl.pointer_type(tl.bfloat16):
                dxk_e = dxk_e.to(tl.bfloat16); dxk_o = dxk_o.to(tl.bfloat16)

            tl.store(row_dxq + idx_e, dxq_e, mask=m)
            tl.store(row_dxq + idx_o, dxq_o, mask=m)
            tl.store(row_dxk + idx_e, dxk_e, mask=m)
            tl.store(row_dxk + idx_o, dxk_o, mask=m)

        for off in range(RD, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            vq = tl.load(row_dyq + idx, mask=m, other=0.0)
            vk = tl.load(row_dyk + idx, mask=m, other=0.0)
            tl.store(row_dxq + idx, vq, mask=m)
            tl.store(row_dxk + idx, vk, mask=m)

    @triton.jit
    def _rope2_of_fwd(XQ, XK, POS, INV, YQ, YK, N, RD, BLOCK: tl.constexpr):
        """On-the-fly RoPE for Q and K using inv_freq and position ids.
        INV has length RD//2 containing frequencies per pair.
        POS has length M (one pos per row).
        """
        pid = tl.program_id(0)
        row_xq = XQ + pid * N
        row_xk = XK + pid * N
        row_yq = YQ + pid * N
        row_yk = YK + pid * N
        pos = tl.load(POS + pid)
        pos_f = pos.to(tl.float32)

        num_pairs = RD // 2
        for p in range(0, tl.cdiv(num_pairs, BLOCK)):
            ip = p * BLOCK + tl.arange(0, BLOCK)
            m = ip < num_pairs
            idx_e = ip * 2
            idx_o = idx_e + 1
            inv = tl.load(INV + ip, mask=m, other=0.0).to(tl.float32)
            ang = pos_f * inv
            c = tl.cos(ang)
            s = tl.sin(ang)

            q_e = tl.load(row_xq + idx_e, mask=m, other=0.0)
            q_o = tl.load(row_xq + idx_o, mask=m, other=0.0)
            k_e = tl.load(row_xk + idx_e, mask=m, other=0.0)
            k_o = tl.load(row_xk + idx_o, mask=m, other=0.0)

            yq_e = q_e.to(tl.float32) * c - q_o.to(tl.float32) * s
            yq_o = q_o.to(tl.float32) * c + q_e.to(tl.float32) * s
            yk_e = k_e.to(tl.float32) * c - k_o.to(tl.float32) * s
            yk_o = k_o.to(tl.float32) * c + k_e.to(tl.float32) * s

            if tl.typeof(YQ) == tl.pointer_type(tl.float16):
                yq_e = yq_e.to(tl.float16); yq_o = yq_o.to(tl.float16)
            elif tl.typeof(YQ) == tl.pointer_type(tl.bfloat16):
                yq_e = yq_e.to(tl.bfloat16); yq_o = yq_o.to(tl.bfloat16)
            if tl.typeof(YK) == tl.pointer_type(tl.float16):
                yk_e = yk_e.to(tl.float16); yk_o = yk_o.to(tl.float16)
            elif tl.typeof(YK) == tl.pointer_type(tl.bfloat16):
                yk_e = yk_e.to(tl.bfloat16); yk_o = yk_o.to(tl.bfloat16)

            tl.store(row_yq + idx_e, yq_e, mask=m)
            tl.store(row_yq + idx_o, yq_o, mask=m)
            tl.store(row_yk + idx_e, yk_e, mask=m)
            tl.store(row_yk + idx_o, yk_o, mask=m)

        for off in range(RD, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            vq = tl.load(row_xq + idx, mask=m, other=0.0)
            vk = tl.load(row_xk + idx, mask=m, other=0.0)
            tl.store(row_yq + idx, vq, mask=m)
            tl.store(row_yk + idx, vk, mask=m)

    @triton.jit
    def _rope2_of_bwd(DYQ, DYK, POS, INV, DXQ, DXK, N, RD, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_dyq = DYQ + pid * N
        row_dyk = DYK + pid * N
        row_dxq = DXQ + pid * N
        row_dxk = DXK + pid * N
        pos = tl.load(POS + pid)
        pos_f = pos.to(tl.float32)

        num_pairs = RD // 2
        for p in range(0, tl.cdiv(num_pairs, BLOCK)):
            ip = p * BLOCK + tl.arange(0, BLOCK)
            m = ip < num_pairs
            idx_e = ip * 2
            idx_o = idx_e + 1
            inv = tl.load(INV + ip, mask=m, other=0.0).to(tl.float32)
            ang = pos_f * inv
            c = tl.cos(ang)
            s = tl.sin(ang)

            dyq_e = tl.load(row_dyq + idx_e, mask=m, other=0.0).to(tl.float32)
            dyq_o = tl.load(row_dyq + idx_o, mask=m, other=0.0).to(tl.float32)
            dyk_e = tl.load(row_dyk + idx_e, mask=m, other=0.0).to(tl.float32)
            dyk_o = tl.load(row_dyk + idx_o, mask=m, other=0.0).to(tl.float32)

            dxq_e = dyq_e * c + dyq_o * s
            dxq_o = -dyq_e * s + dyq_o * c
            dxk_e = dyk_e * c + dyk_o * s
            dxk_o = -dyk_e * s + dyk_o * c

            if tl.typeof(DXQ) == tl.pointer_type(tl.float16):
                dxq_e = dxq_e.to(tl.float16); dxq_o = dxq_o.to(tl.float16)
            elif tl.typeof(DXQ) == tl.pointer_type(tl.bfloat16):
                dxq_e = dxq_e.to(tl.bfloat16); dxq_o = dxq_o.to(tl.bfloat16)
            if tl.typeof(DXK) == tl.pointer_type(tl.float16):
                dxk_e = dxk_e.to(tl.float16); dxk_o = dxk_o.to(tl.float16)
            elif tl.typeof(DXK) == tl.pointer_type(tl.bfloat16):
                dxk_e = dxk_e.to(tl.bfloat16); dxk_o = dxk_o.to(tl.bfloat16)

            tl.store(row_dxq + idx_e, dxq_e, mask=m)
            tl.store(row_dxq + idx_o, dxq_o, mask=m)
            tl.store(row_dxk + idx_e, dxk_e, mask=m)
            tl.store(row_dxk + idx_o, dxk_o, mask=m)

        for off in range(RD, N, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            m = idx < N
            vq = tl.load(row_dyq + idx, mask=m, other=0.0)
            vk = tl.load(row_dyk + idx, mask=m, other=0.0)
            tl.store(row_dxq + idx, vq, mask=m)
            tl.store(row_dxk + idx, vk, mask=m)


class _TritonRoPEFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int):
        if not (x.is_cuda and _has_triton()):
            # CPU or non-Triton path: pure torch fallback
            return _rope_torch(x, cos, sin, rotary_dim)

        B = x.shape[0]
        N = x.shape[-1]
        RD = rotary_dim

        x_2d = x.contiguous().view(B, N)
        y = torch.empty_like(x_2d)
        cos_2d = cos.contiguous().view(B, RD)
        sin_2d = sin.contiguous().view(B, RD)

        BLOCK = 128
        grid = (B,)
        _rope_fwd[grid](x_2d, cos_2d, sin_2d, y, N, RD, BLOCK=BLOCK)
        ctx.save_for_backward(cos_2d, sin_2d)
        ctx.shape = x.shape
        ctx.N = N
        ctx.RD = RD
        return y.view_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (cos_2d, sin_2d) = ctx.saved_tensors
        N, RD = ctx.N, ctx.RD
        B = dy.numel() // N
        dy_2d = dy.contiguous().view(B, N)
        dx = torch.empty_like(dy_2d)

        if dy.is_cuda and _has_triton():
            BLOCK = 128
            grid = (B,)
            _rope_bwd[grid](dy_2d, cos_2d, sin_2d, dx, N, RD, BLOCK=BLOCK)
        else:
            # Torch fallback for backward
            dx[:] = _rope_torch_bwd(dy_2d, cos_2d, sin_2d, RD)

        return dx.view(ctx.shape), None, None, None


class _TritonRoPE2Fn(Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int):
        if not (q.is_cuda and k.is_cuda and _has_triton()):
            return _rope_torch(q, cos, sin, rotary_dim), _rope_torch(k, cos, sin, rotary_dim)
        B = q.shape[0]
        N = q.shape[-1]
        RD = rotary_dim
        yq = torch.empty_like(q)
        yk = torch.empty_like(k)
        BLOCK = 128
        grid = (B,)
        _rope2_fwd[grid](q, k, cos, sin, yq, yk, N, RD, BLOCK=BLOCK)
        ctx.save_for_backward(cos, sin)
        ctx.shape = q.shape
        ctx.N = N
        ctx.RD = RD
        return yq, yk

    @staticmethod
    def backward(ctx, dyq: torch.Tensor, dyk: torch.Tensor):
        (cos_2d, sin_2d) = ctx.saved_tensors
        N, RD = ctx.N, ctx.RD
        B = dyq.shape[0]
        dxq = torch.empty_like(dyq)
        dxk = torch.empty_like(dyk)
        if dyq.is_cuda and dyk.is_cuda and _has_triton():
            BLOCK = 128
            grid = (B,)
            _rope2_bwd[grid](dyq, dyk, cos_2d, sin_2d, dxq, dxk, N, RD, BLOCK=BLOCK)
        else:
            dxq[:] = _rope_torch_bwd(dyq, cos_2d, sin_2d, RD)
            dxk[:] = _rope_torch_bwd(dyk, cos_2d, sin_2d, RD)
        return dxq, dxk, None, None, None


class _TritonRoPE2OnFlyFn(Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, pos_flat: torch.Tensor, inv_freq: torch.Tensor, rotary_dim: int):
        if not (q.is_cuda and k.is_cuda and _has_triton()):
            # CPU fallback using explicit cos/sin compute
            bsxh = pos_flat.numel() // q.shape[1] if q.dim() == 2 else None
            raise RuntimeError("On-the-fly CPU path not implemented; expected CUDA Triton for AZR_ROPE_EPI.")
        B = q.shape[0]
        N = q.shape[-1]
        RD = rotary_dim
        yq = torch.empty_like(q)
        yk = torch.empty_like(k)
        BLOCK = 128
        grid = (B,)
        _rope2_of_fwd[grid](q, k, pos_flat, inv_freq, yq, yk, N, RD, BLOCK=BLOCK)
        ctx.save_for_backward(pos_flat, inv_freq)
        ctx.shape = q.shape
        ctx.N = N
        ctx.RD = RD
        return yq, yk

    @staticmethod
    def backward(ctx, dyq: torch.Tensor, dyk: torch.Tensor):
        (pos_flat, inv_freq) = ctx.saved_tensors
        N, RD = ctx.N, ctx.RD
        B = dyq.shape[0]
        dxq = torch.empty_like(dyq)
        dxk = torch.empty_like(dyk)
        if dyq.is_cuda and dyk.is_cuda and _has_triton():
            BLOCK = 128
            grid = (B,)
            _rope2_of_bwd[grid](dyq, dyk, pos_flat, inv_freq, dxq, dxk, N, RD, BLOCK=BLOCK)
        else:
            raise RuntimeError("On-the-fly CPU backward not implemented; expected CUDA.")
        return dxq, dxk, None, None, None


def _rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Torch reference: y = x * cos + rotate_half(x) * sin on the first rotary_dim dims.
    cos/sin expected shape broadcastable to x over last dim.
    """
    d = x.size(-1)
    d_rot = rotary_dim
    x1 = x[..., :d_rot]
    x2 = x[..., d_rot:]
    # rotate_half
    x1_e = x1[..., 0::2]
    x1_o = x1[..., 1::2]
    # Apply elementwise cos/sin
    ce = cos[..., 0::2]
    co = cos[..., 1::2]
    se = sin[..., 0::2]
    so = sin[..., 1::2]
    y1_e = x1_e * ce - x1_o * se
    y1_o = x1_o * co + x1_e * so
    y1 = torch.empty_like(x1)
    y1[..., 0::2] = y1_e
    y1[..., 1::2] = y1_o
    return torch.cat([y1, x2], dim=-1)


def _rope_torch_bwd(dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Torch backward reference for RoPE when Triton is unavailable.
    dx_even = dy_even * cos_even + dy_odd * sin_odd
    dx_odd  = -dy_even * sin_even + dy_odd * cos_odd
    """
    d = dy.size(-1)
    d_rot = rotary_dim
    dy1 = dy[..., :d_rot]
    dy2 = dy[..., d_rot:]

    dy1_e = dy1[..., 0::2]
    dy1_o = dy1[..., 1::2]
    ce = cos[..., 0::2]
    co = cos[..., 1::2]
    se = sin[..., 0::2]
    so = sin[..., 1::2]

    dx1_e = dy1_e * ce + dy1_o * so
    dx1_o = -dy1_e * se + dy1_o * co
    dx1 = torch.empty_like(dy1)
    dx1[..., 0::2] = dx1_e
    dx1[..., 1::2] = dx1_o
    return torch.cat([dx1, dy2], dim=-1)


def _expand_and_gather_cos_sin(cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, target_shape: torch.Size, rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct cos/sin tensors aligned per-token for the target shape.
    Returns flattened 2D tensors [M, rotary_dim] aligned to rows.
    Supports typical HF cache shapes.
    """
    device = position_ids.device
    bs, sl = position_ids.shape
    rd = rotary_dim

    # Try common shapes:
    # 1) [seq, dim]
    if cos.dim() == 2 and cos.size(0) >= sl and cos.size(1) >= rd:
        cos_sel = cos.index_select(0, position_ids.reshape(-1)).reshape(bs, sl, rd)
        sin_sel = sin.index_select(0, position_ids.reshape(-1)).reshape(bs, sl, rd)
    # 2) [1, 1, seq, dim] or [bs, 1, seq, dim]
    elif cos.dim() == 4 and cos.size(-1) >= rd:
        # gather along seq dim = -2
        idx = position_ids.view(bs, 1, sl, 1).expand(bs, 1, sl, rd)
        # If batch dimension is 1, expand; otherwise rely on matching bs
        if cos.size(0) == 1:
            cos_sel = cos.expand(bs, -1, -1, -1).gather(-2, idx).squeeze(1)
            sin_sel = sin.expand(bs, -1, -1, -1).gather(-2, idx).squeeze(1)
        else:
            cos_sel = cos.gather(-2, idx).squeeze(1)
            sin_sel = sin.gather(-2, idx).squeeze(1)
    # 3) [seq, 1, dim] (rare)
    elif cos.dim() == 3 and cos.size(0) >= sl and cos.size(-1) >= rd:
        cos_sel = cos.index_select(0, position_ids.reshape(-1)).reshape(bs, sl, rd)
        sin_sel = sin.index_select(0, position_ids.reshape(-1)).reshape(bs, sl, rd)
    else:
        # As a last resort, try simple indexing
        try:
            cos_sel = cos[position_ids]
            sin_sel = sin[position_ids]
        except Exception as e:
            raise RuntimeError(f"Unsupported cos/sin shapes for RoPE fastpath: cos={cos.shape}, sin={sin.shape}") from e

    # Ensure dtype/device alignment
    cos_sel = cos_sel.to(device=device, dtype=torch.float32)
    sin_sel = sin_sel.to(device=device, dtype=torch.float32)

    # Broadcast to match target without head dim, then expand over heads later
    # target_shape expected: [..., heads, dim]
    # We only need [bs, sl, rd] here.
    return cos_sel[..., :rd].contiguous(), sin_sel[..., :rd].contiguous()


def _apply_rope_fast(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply fused RoPE via Triton to q and k.
    Supports q/k shapes [bs, sl, h, d] or [bs*h, sl, d].
    """
    # Normalize shapes to [M, N]
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    if q.dim() == 4:
        bs, sl, h, d = q.shape
        M = bs * sl * h
        q_flat = q.contiguous().view(M, d)
        k_flat = k.contiguous().view(M, d)
        cos_sel, sin_sel = _expand_and_gather_cos_sin(cos, sin, position_ids, q.shape, rotary_dim)
        cos_flat = cos_sel.unsqueeze(2).expand(bs, sl, h, rotary_dim).contiguous().view(M, rotary_dim)
        sin_flat = sin_sel.unsqueeze(2).expand(bs, sl, h, rotary_dim).contiguous().view(M, rotary_dim)
    elif q.dim() == 3:
        # [bs*h, sl, d]
        bh, sl, d = q.shape
        M = bh * sl
        q_flat = q.contiguous().view(M, d)
        k_flat = k.contiguous().view(M, d)
        # Need bs to compute cos; infer bs from position_ids
        bs = position_ids.size(0)
        h = bh // bs
        cos_sel, sin_sel = _expand_and_gather_cos_sin(cos, sin, position_ids, q.shape, rotary_dim)
        cos_flat = cos_sel.unsqueeze(2).expand(bs, sl, h, rotary_dim).contiguous().view(M, rotary_dim)
        sin_flat = sin_sel.unsqueeze(2).expand(bs, sl, h, rotary_dim).contiguous().view(M, rotary_dim)
    else:
        raise RuntimeError(f"Unsupported q shape for RoPE fastpath: {q.shape}")

    # Autograd-enabled fused application: prefer dual-input fused kernel
    if q_flat.is_cuda and _has_triton():
        yq, yk = _TritonRoPE2Fn.apply(q_flat, k_flat, cos_flat, sin_flat, rotary_dim)
    else:
        yq = _rope_torch(q_flat, cos_flat, sin_flat, rotary_dim)
        yk = _rope_torch(k_flat, cos_flat, sin_flat, rotary_dim)

    return yq.view(orig_q_shape), yk.view(orig_k_shape)


def try_patch_rope(model: nn.Module) -> None:
    """Monkey-patch Transformers' apply_rotary_pos_emb with a Triton fast path.
    Enabled when AZR_TRITON_ROPE=1 and Triton is available.
    Falls back to the original for unsupported shapes/dtypes/devices.
    """
    if os.environ.get("AZR_TRITON_ROPE", "0") != "1":
        return
    if not _has_triton():
        warnings.warn("AZR_TRITON_ROPE=1 but Triton is not available; skipping.")
        return

    try:
        import transformers.models.llama.modeling_llama as modeling_llama
    except Exception:
        warnings.warn("RoPE fastpath currently supports LLaMA family only.")
        return

    if not hasattr(modeling_llama, "apply_rotary_pos_emb"):
        return

    orig_apply = modeling_llama.apply_rotary_pos_emb

    # Optionally capture inv_freq from LLaMA rotary embedding for on-the-fly kernel
    if os.environ.get("AZR_ROPE_EPI", "0") == "1":
        try:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

            orig_rotary_forward = LlamaRotaryEmbedding.forward

            def cap_forward(self, x, *args, **kwargs):
                out = orig_rotary_forward(self, x, *args, **kwargs)
                try:
                    set_last_inv_freq(self.inv_freq.to(x.device))
                except Exception:
                    pass
                return out

            LlamaRotaryEmbedding.forward = cap_forward
        except Exception:
            warnings.warn("AZR_ROPE_EPI=1 set, but failed to patch LlamaRotaryEmbedding for inv_freq capture.")

    def azr_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1, rotary_dim=None):
        # Determine rotary_dim from inputs if not provided; default to full head dim
        d = q.size(-1)
        rd = rotary_dim if isinstance(rotary_dim, int) and 0 < rotary_dim <= d else d
        try:
            # Use fast path only for CUDA tensors with supported dtypes
            if q.is_cuda and k.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype:
                # On-the-fly epilogue path if enabled and inv_freq captured
                if os.environ.get("AZR_ROPE_EPI", "0") == "1" and _ROPE_LAST_INV_FREQ is not None:
                    # Flatten shapes to [M, N]
                    if q.dim() == 4:
                        bs, sl, h, d_ = q.shape
                        pos_flat = position_ids.view(bs, sl, 1).expand(bs, sl, h).contiguous().view(-1)
                        M = bs * sl * h
                        q_flat = q.contiguous().view(M, d_)
                        k_flat = k.contiguous().view(M, d_)
                    elif q.dim() == 3:
                        # [bs*h, sl, d]
                        bh, sl, d_ = q.shape
                        # infer bs from position_ids
                        bs = position_ids.size(0)
                        h = bh // bs
                        pos_flat = position_ids.view(bs, sl, 1).expand(bs, sl, h).contiguous().view(-1)
                        M = bh * sl
                        q_flat = q.contiguous().view(M, d_)
                        k_flat = k.contiguous().view(M, d_)
                    else:
                        return orig_apply(q, k, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim)

                    inv = _ROPE_LAST_INV_FREQ[: rd // 2].contiguous().to(device=q.device)
                    yq, yk = _TritonRoPE2OnFlyFn.apply(q_flat, k_flat, pos_flat, inv, rd)
                    return yq.view_as(q), yk.view_as(k)
                else:
                    yq, yk = _apply_rope_fast(q, k, cos, sin, position_ids, rd)
                    return yq, yk
        except Exception:
            # Fall back on any failure
            pass
        return orig_apply(q, k, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim)

    # Patch module-level function
    try:
        modeling_llama.apply_rotary_pos_emb = azr_apply_rotary_pos_emb
    except Exception as e:
        warnings.warn(f"Failed to patch apply_rotary_pos_emb: {e}")
