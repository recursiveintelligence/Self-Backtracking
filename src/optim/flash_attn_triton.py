import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def _flash_attn_fwd(Q, K, V, O, MBUF, LBUF, BHM, S, D, SCALE, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr):
        """
        FlashAttention forward pass without dropout. Shapes:
        - Q, K, V, O: [B*H, S, D] in row-major contiguous layout.
        - BHM: number of (batch*heads)
        - SCALE: 1/sqrt(D) (float)
        - CAUSAL: bool constexpr to enable causal masking.
        """
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        # Bounds check
        if pid_bh >= BHM:
            return

        # Offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptr = Q + pid_bh * S * D
        k_ptr = K + pid_bh * S * D
        v_ptr = V + pid_bh * S * D
        o_ptr = O + pid_bh * S * D

        # Load Q tile [BLOCK_M, D]
        q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :], mask=(offs_m[:, None] < S) & (offs_d[None, :] < D), other=0.0)
        q = q.to(tl.float32) * SCALE

        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # Iterate over K/V tiles
        for start_n in range(0, S, BLOCK_N):
            n_idx = start_n + offs_n
            # Load K [BLOCK_N, D] and V [BLOCK_N, D]
            k = tl.load(k_ptr + n_idx[:, None] * D + offs_d[None, :], mask=(n_idx[:, None] < S) & (offs_d[None, :] < D), other=0.0)
            v = tl.load(v_ptr + n_idx[:, None] * D + offs_d[None, :], mask=(n_idx[:, None] < S) & (offs_d[None, :] < D), other=0.0)
            k = k.to(tl.float32)
            v = v.to(tl.float32)

            # Compute attention scores [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, tl.trans(k))  # fp32

            # Causal mask: disallow attending to future positions
            if CAUSAL:
                # i in offs_m, j in n_idx: mask where j > i
                i_broad = offs_m[:, None]
                j_broad = n_idx[None, :]
                causal_mask = j_broad > i_broad
                qk = tl.where(causal_mask, float('-inf'), qk)

            # Numerically stable softmax update
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_i_new[:, None])
            l_i_new = l_i * tl.exp(m_i - m_i_new) + tl.sum(p, axis=1)
            # Attention * V accumulate
            acc = acc * (l_i / l_i_new)[:, None] + tl.dot(p, v)
            m_i = m_i_new
            l_i = l_i_new

        # Normalize and write O
        o = acc / l_i[:, None]
        # Cast back to output dtype
        o = o.to(tl.float16) if tl.typeof(O) == tl.pointer_type(tl.float16) else (o.to(tl.bfloat16) if tl.typeof(O) == tl.pointer_type(tl.bfloat16) else o)
        tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], o, mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))
        # Store per-row m and l for backward
        tl.store(MBUF + pid_bh * S + offs_m, m_i, mask=(offs_m < S))
        tl.store(LBUF + pid_bh * S + offs_m, l_i, mask=(offs_m < S))

    @triton.jit
    def _flash_attn_bwd(Q, K, V, DO, MBUF, LBUF, DQ, DK, DV, BHM, S, D, SCALE, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr):
        """Backward pass with two-phase streaming over K/V tiles.
        Accumulates dK and dV via atomics. DQ is written directly by row block.
        """
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        if pid_bh >= BHM:
            return
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptr = Q + pid_bh * S * D
        k_ptr = K + pid_bh * S * D
        v_ptr = V + pid_bh * S * D
        do_ptr = DO + pid_bh * S * D
        dq_ptr = DQ + pid_bh * S * D
        dk_ptr = DK + pid_bh * S * D
        dv_ptr = DV + pid_bh * S * D

        # Load Q and dO tiles [BM, D]
        q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :], mask=(offs_m[:, None] < S) & (offs_d[None, :] < D), other=0.0).to(tl.float32) * SCALE
        dO = tl.load(do_ptr + offs_m[:, None] * D + offs_d[None, :], mask=(offs_m[:, None] < S) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # Load m and l
        m_i = tl.load(MBUF + pid_bh * S + offs_m, mask=(offs_m < S), other=float('-inf'))
        l_i = tl.load(LBUF + pid_bh * S + offs_m, mask=(offs_m < S), other=1.0)

        # First pass: accumulate sum_dp = sum_j p_ij * dp_ij
        sum_dp = tl.zeros([BLOCK_M], dtype=tl.float32)
        for start_n in range(0, S, BLOCK_N):
            n_idx = start_n + offs_n
            k = tl.load(k_ptr + n_idx[:, None] * D + offs_d[None, :], mask=(n_idx[:, None] < S) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
            v = tl.load(v_ptr + n_idx[:, None] * D + offs_d[None, :], mask=(n_idx[:, None] < S) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
            qk = tl.dot(q, tl.trans(k))
            if CAUSAL:
                i_broad = offs_m[:, None]
                j_broad = n_idx[None, :]
                causal_mask = j_broad > i_broad
                qk = tl.where(causal_mask, float('-inf'), qk)
            p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
            dp = tl.dot(dO, tl.trans(v))
            sum_dp += tl.sum(p * dp, axis=1)

        # Second pass: compute gradients
        dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        for start_n in range(0, S, BLOCK_N):
            n_idx = start_n + offs_n
            k = tl.load(k_ptr + n_idx[:, None] * D + offs_d[None, :], mask=(n_idx[:, None] < S) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
            v = tl.load(v_ptr + n_idx[:, None] * D + offs_d[None, :], mask=(n_idx[:, None] < S) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
            qk = tl.dot(q, tl.trans(k))
            if CAUSAL:
                i_broad = offs_m[:, None]
                j_broad = n_idx[None, :]
                causal_mask = j_broad > i_broad
                qk = tl.where(causal_mask, float('-inf'), qk)
            p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
            dp = tl.dot(dO, tl.trans(v))
            ds = p * (dp - sum_dp[:, None])
            # dQ accumulate: ds @ K (and account for scale)
            dq_acc += tl.dot(ds, k)
            # dK atomic add: ds^T @ Q
            dk_tile = tl.dot(tl.trans(ds), q)
            # Undo scale for dK (since q was scaled): d/dK uses q*scale
            # But since q already included scale in qk, gradients are consistent.
            # dV atomic add: p^T @ dO
            dv_tile = tl.dot(tl.trans(p), dO)

            # Atomically accumulate into DK/DV
            tl.atomic_add(dk_ptr + n_idx[:, None] * D + offs_d[None, :], dk_tile, mask=(n_idx[:, None] < S) & (offs_d[None, :] < D))
            tl.atomic_add(dv_ptr + n_idx[:, None] * D + offs_d[None, :], dv_tile, mask=(n_idx[:, None] < S) & (offs_d[None, :] < D))

        # Store dQ (undo scale factor)
        dq_acc = dq_acc * SCALE  # derivative through scaled q
        dq_out = dq_acc.to(tl.float16) if tl.typeof(DQ) == tl.pointer_type(tl.float16) else (dq_acc.to(tl.bfloat16) if tl.typeof(DQ) == tl.pointer_type(tl.bfloat16) else dq_acc)
        tl.store(dq_ptr + offs_m[:, None] * D + offs_d[None, :], dq_out, mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))


class _FlashAttnTritonFn(Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        assert q.is_cuda and k.is_cuda and v.is_cuda and _has_triton(), "FlashAttn Triton requires CUDA + Triton"
        BH, S, D = q.shape
        o = torch.empty_like(q)
        # Choose tile sizes conservatively
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(128, ((D + 31) // 32) * 32)
        grid = (BH, triton.cdiv(S, BLOCK_M))
        scale = (1.0 / (D ** 0.5))
        # Buffers for backward
        m = torch.empty((BH, S), device=q.device, dtype=torch.float32)
        l = torch.empty((BH, S), device=q.device, dtype=torch.float32)
        _flash_attn_fwd[grid](
            q, k, v, o, m, l,
            BH, S, D, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_D,
            CAUSAL=causal,
        )
        ctx.save_for_backward(q, k, v, m, l)
        ctx.causal = causal
        ctx.tile = (BLOCK_M, BLOCK_N, BLOCK_D)
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        # Triton backward using saved m and l
        q, k, v, m, l = ctx.saved_tensors
        BH, S, D = q.shape
        BLOCK_M, BLOCK_N, BLOCK_D = ctx.tile
        grid = (BH, triton.cdiv(S, BLOCK_M))
        dq = torch.empty_like(q)
        # Accumulate dK/dV in fp32 and cast on return
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        scale = (1.0 / (D ** 0.5))
        _flash_attn_bwd[grid](
            q, k, v, do, m, l, dq, dk, dv,
            BH, S, D, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_D,
            CAUSAL=ctx.causal,
        )
        return dq, dk.to(k.dtype), dv.to(v.dtype), None


def flash_attn_triton(q_bhsd: torch.Tensor, k_bhsd: torch.Tensor, v_bhsd: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Compute attention via Triton FlashAttention forward, autograd-compatible backward (math fallback).
    Inputs: [B*H, S, D] in half/bf16.
    Returns: [B*H, S, D]
    """
    return _FlashAttnTritonFn.apply(q_bhsd, k_bhsd, v_bhsd, causal)


def try_patch_llama_flash_rope(model: nn.Module) -> None:
    """
    Patch LLaMA attention forward to use:
    - Fused QKV (already patched elsewhere when enabled)
    - On-the-fly fused RoPE for Q/K (from rope_triton)
    - Triton FlashAttention forward with math backward
    Enabled by AZR_TRITON_FLASH_ROPE=1; falls back to original forward on unsupported cases.
    """
    if os.environ.get("AZR_TRITON_FLASH_ROPE", "0") != "1":
        return
    if not _has_triton():
        warnings.warn("AZR_TRITON_FLASH_ROPE=1 but Triton is not available; skipping.")
        return
    try:
        import transformers.models.llama.modeling_llama as modeling_llama
        from .rope_triton import _ROPE_LAST_INV_FREQ, _TritonRoPE2OnFlyFn
    except Exception:
        warnings.warn("Flash+RoPE patch currently supports LLaMA family only.")
        return

    LlamaAttention = getattr(modeling_llama, 'LlamaAttention', None)
    if LlamaAttention is None:
        return

    def _patched_forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False, **kwargs):
        # Fallback conditions
        if (not hidden_states.is_cuda) or (hidden_states.dtype not in (torch.float16, torch.bfloat16)):
            return _orig_forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        if past_key_value is not None or use_cache:
            return _orig_forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        if attention_mask is not None:
            # Only pure causal supported here
            return _orig_forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        if _ROPE_LAST_INV_FREQ is None:
            # Need inv_freq capture via AZR_ROPE_EPI
            return _orig_forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)

        bsz, q_len, _ = hidden_states.size()
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [B, S, H, D]
        def shape(x):
            return x.view(bsz, q_len, num_heads, head_dim)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        # Apply RoPE on-the-fly using inv_freq and position ids
        rd = head_dim  # standard LLaMA uses full head dim for rotary
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, q_len)

        # Build flattened shapes for RoPE on-the-fly function
        bh = bsz * num_heads
        q_flat = q.contiguous().view(bh, q_len, head_dim)
        k_flat = k.contiguous().view(bh, q_len, head_dim)
        pos_flat = position_ids.view(bsz, q_len, 1).expand(bsz, q_len, num_heads).contiguous().view(-1)
        inv = _ROPE_LAST_INV_FREQ[: rd // 2].contiguous().to(device=hidden_states.device)

        yq, yk = _TritonRoPE2OnFlyFn.apply(q_flat, k_flat, pos_flat, inv, rd)

        # Flash attention expects [BH, S, D]
        yq = yq.contiguous()
        yk = yk.contiguous()
        v_flat = v.contiguous().view(bh, q_len, head_dim)

        o = flash_attn_triton(yq, yk, v_flat, causal=True)

        # Reshape back and output projection
        o = o.view(bsz, num_heads, q_len, head_dim).transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        out = self.o_proj(o)
        if output_attentions:
            return out, None, None
        return out, None

    # Bind patched forward to attention modules
    count = 0
    for attn in model.modules():
        if isinstance(attn, LlamaAttention):
            if getattr(attn, "_azr_flash_rope_patched", False):
                continue
            _orig_forward = attn.forward
            attn.forward = _patched_forward.__get__(attn, LlamaAttention)  # type: ignore
            attn._azr_flash_rope_patched = True
            count += 1
    if count == 0:
        warnings.warn("No LLaMA attention modules patched for Flash+RoPE.")
