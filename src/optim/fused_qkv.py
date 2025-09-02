import os
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn


class _QKVFuserCore(nn.Module):
    """Holds references to q_proj, k_proj, v_proj and exposes a fused compute.
    It monkey-patches the forward of q/k/v to call this fused path, avoiding 3 GEMMs per attention.
    Parameters remain owned by the original Linear modules to keep optimizer compatibility.
    """

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self._cache_input_ptr: Optional[int] = None
        self._cache_q: Optional[torch.Tensor] = None
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def _compute(self, x: torch.Tensor):
        # Build concatenated weights/biases dynamically to keep grads flowing to each module separately.
        W = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        b = None
        if self.q_proj.bias is not None and self.k_proj.bias is not None and self.v_proj.bias is not None:
            b = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)
        y = F.linear(x, W, b)
        sizes = [self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features]
        dq, dk, dv = torch.split(y, sizes, dim=-1)
        return dq, dk, dv

    def get(self, x: torch.Tensor, which: str) -> torch.Tensor:
        ip = x.data_ptr() if x.is_cuda else id(x)
        if self._cache_input_ptr != ip or self._cache_q is None:
            q, k, v = self._compute(x)
            self._cache_input_ptr = ip
            self._cache_q, self._cache_k, self._cache_v = q, k, v
        if which == 'q':
            return self._cache_q
        if which == 'k':
            return self._cache_k
        if which == 'v':
            return self._cache_v
        raise ValueError(which)


def try_patch_llama_fused_qkv(model: nn.Module) -> None:
    """
    Enable fused QKV projection for LLaMA attention by monkey-patching q/k/v forward methods
    to route through a single GEMM. Gated by AZR_FUSE_QKV=1.
    """
    if os.environ.get("AZR_FUSE_QKV", "0") != "1":
        return
    try:
        import transformers.models.llama.modeling_llama as modeling_llama
        LlamaAttention = modeling_llama.LlamaAttention
    except Exception:
        warnings.warn("Fused QKV patch currently supports LLaMA family only.")
        return

    count = 0
    for attn in model.modules():
        if isinstance(attn, LlamaAttention):
            # Skip if already patched
            if getattr(attn, "_azr_qkv_patched", False):
                continue
            q_proj = getattr(attn, "q_proj", None)
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            if not all(isinstance(m, nn.Linear) for m in (q_proj, k_proj, v_proj)):
                continue

            core = _QKVFuserCore(q_proj, k_proj, v_proj)

            def q_forward(self, x):
                return core.get(x, 'q')

            def k_forward(self, x):
                return core.get(x, 'k')

            def v_forward(self, x):
                return core.get(x, 'v')

            # Bind the new forward methods
            q_proj.forward = q_forward.__get__(q_proj, nn.Linear)  # type: ignore[attr-defined]
            k_proj.forward = k_forward.__get__(k_proj, nn.Linear)  # type: ignore[attr-defined]
            v_proj.forward = v_forward.__get__(v_proj, nn.Linear)  # type: ignore[attr-defined]

            # Mark patched
            attn._azr_qkv_patched = True
            count += 1
    if count == 0:
        warnings.warn("No LLaMA attention modules patched for fused QKV.")
