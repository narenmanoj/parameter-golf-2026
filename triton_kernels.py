"""
Fused Triton kernels for the parameter-golf transformer.

Kernel 1: fused_residmix_rmsnorm  — Block preamble (mix + RMSNorm)
Kernel 2: fused_residadd_rmsnorm  — Post-attention residual add + scale + RMSNorm for MLP input
Kernel 3: fused_smeargate_rmsnorm — SmearGate + RMSNorm (embedding pipeline)
"""

from __future__ import annotations

import torch
import torch.nn.functional  # for pad in backward
import triton
import triton.language as tl
from torch import Tensor


# -------------------------------------------------------------------------
# Kernel 1: fused residual mixing + RMSNorm
#   y_i = mix0_i * x_i + mix1_i * x0_i
#   out_i = y_i / rms(y)
# Each program handles one (batch, seq) row of width D.
# -------------------------------------------------------------------------

@triton.jit
def _fused_residmix_rmsnorm_fwd(
    X_ptr, X0_ptr, Mix0_ptr, Mix1_ptr, Out_ptr, Mixed_ptr,
    stride_bst,  # stride between rows (B*T dimension)
    D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(X_ptr + row * stride_bst + offs, mask=offs < D).to(tl.float32)
    x0 = tl.load(X0_ptr + row * stride_bst + offs, mask=offs < D).to(tl.float32)
    m0 = tl.load(Mix0_ptr + offs, mask=offs < D).to(tl.float32)
    m1 = tl.load(Mix1_ptr + offs, mask=offs < D).to(tl.float32)

    # Residual mix
    y = m0 * x + m1 * x0

    # Save mixed output (needed for backward through mix params)
    tl.store(Mixed_ptr + row * stride_bst + offs, y.to(tl.bfloat16), mask=offs < D)

    # RMSNorm
    var = tl.sum(y * y, axis=0) / D
    rrms = 1.0 / tl.sqrt(var + eps)
    out = y * rrms

    tl.store(Out_ptr + row * stride_bst + offs, out.to(tl.bfloat16), mask=offs < D)


class FusedResidMixRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, x0: Tensor, mix0: Tensor, mix1: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        B_T = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        x0_flat = x0.reshape(-1, D)

        out = torch.empty_like(x_flat)
        mixed = torch.empty_like(x_flat)

        # Ensure D is power-of-2 for Triton (pad if needed)
        assert D <= 8192, f"D={D} too large for Triton kernel"

        _fused_residmix_rmsnorm_fwd[(B_T,)](
            x_flat, x0_flat,
            mix0.to(x.dtype), mix1.to(x.dtype),
            out, mixed,
            stride_bst=D,
            D=triton.next_power_of_2(D),
            eps=eps,
        )

        ctx.save_for_backward(x_flat, x0_flat, mixed, mix0, mix1)
        ctx.eps = eps
        ctx.shape = x.shape
        return mixed.view(x.shape), out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_mixed: Tensor, grad_out: Tensor):
        x_flat, x0_flat, mixed, mix0, mix1 = ctx.saved_tensors
        D = x_flat.shape[-1]

        # Manual RMSNorm backward (avoids requires_grad_ for torch.compile compat)
        mixed_f = mixed.float()
        var = (mixed_f * mixed_f).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + ctx.eps)
        normed = mixed_f * rrms
        grad_out_flat = grad_out.reshape(-1, D).float()
        # d/dx RMSNorm: (grad - normed * mean(grad * normed)) * rrms
        grad_mixed_from_norm = (grad_out_flat - normed * (grad_out_flat * normed).mean(dim=-1, keepdim=True)) * rrms

        # Combine gradients: mixed gets grad from both paths
        total_grad_mixed = grad_mixed_from_norm + grad_mixed.reshape(-1, D).float()

        m0 = mix0.float()
        m1 = mix1.float()
        grad_x = (total_grad_mixed * m0).to(x_flat.dtype).view(ctx.shape)
        grad_x0 = (total_grad_mixed * m1).to(x_flat.dtype).view(ctx.shape)
        grad_mix0 = (total_grad_mixed * x_flat.float()).sum(dim=0)
        grad_mix1 = (total_grad_mixed * x0_flat.float()).sum(dim=0)

        return grad_x, grad_x0, grad_mix0, grad_mix1, None


def fused_residmix_rmsnorm(x: Tensor, x0: Tensor, mix0: Tensor, mix1: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Fused residual mix + RMSNorm. Returns (mixed, normed)."""
    return FusedResidMixRMSNorm.apply(x, x0, mix0, mix1, eps)


# -------------------------------------------------------------------------
# Kernel 2: fused residual-add with scale + RMSNorm
#   x_new = x + scale * attn_out
#   normed = rms_norm(x_new)
# Returns both x_new (for later residual) and normed (for MLP input).
# -------------------------------------------------------------------------

@triton.jit
def _fused_residadd_rmsnorm_fwd(
    X_ptr, Attn_ptr, Scale_ptr, XNew_ptr, Normed_ptr,
    stride_bst,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(X_ptr + row * stride_bst + offs, mask=offs < D).to(tl.float32)
    a = tl.load(Attn_ptr + row * stride_bst + offs, mask=offs < D).to(tl.float32)
    s = tl.load(Scale_ptr + offs, mask=offs < D).to(tl.float32)

    # Residual add with scale
    x_new = x + s * a
    tl.store(XNew_ptr + row * stride_bst + offs, x_new.to(tl.bfloat16), mask=offs < D)

    # RMSNorm
    var = tl.sum(x_new * x_new, axis=0) / D
    rrms = 1.0 / tl.sqrt(var + eps)
    normed = x_new * rrms
    tl.store(Normed_ptr + row * stride_bst + offs, normed.to(tl.bfloat16), mask=offs < D)


class FusedResidAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, attn_out: Tensor, scale: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        B_T = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        attn_flat = attn_out.reshape(-1, D)

        x_new = torch.empty_like(x_flat)
        normed = torch.empty_like(x_flat)

        _fused_residadd_rmsnorm_fwd[(B_T,)](
            x_flat, attn_flat,
            scale.to(x.dtype),
            x_new, normed,
            stride_bst=D,
            D=triton.next_power_of_2(D),
            eps=eps,
        )

        ctx.save_for_backward(x_flat, attn_flat, x_new, scale)
        ctx.eps = eps
        ctx.shape = x.shape
        return x_new.view(x.shape), normed.view(x.shape)

    @staticmethod
    def backward(ctx, grad_x_new: Tensor, grad_normed: Tensor):
        x_flat, attn_flat, x_new, scale = ctx.saved_tensors
        D = x_flat.shape[-1]

        # Manual RMSNorm backward (avoids requires_grad_ for torch.compile compat)
        x_new_f = x_new.float()
        var = (x_new_f * x_new_f).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + ctx.eps)
        normed_recomp = x_new_f * rrms
        grad_normed_flat = grad_normed.reshape(-1, D).float()
        grad_x_new_from_norm = (grad_normed_flat - normed_recomp * (grad_normed_flat * normed_recomp).mean(dim=-1, keepdim=True)) * rrms

        # Combine grads: x_new gets grad from both outputs
        total_grad_x_new = grad_x_new_from_norm + grad_x_new.reshape(-1, D).float()

        # Backward through x_new = x + scale * attn_out
        s = scale.float()
        grad_x = total_grad_x_new.to(x_flat.dtype).view(ctx.shape)
        grad_attn = (total_grad_x_new * s[None, :]).to(attn_flat.dtype).view(ctx.shape)
        grad_scale = (total_grad_x_new * attn_flat.float()).sum(dim=0)

        return grad_x, grad_attn, grad_scale, None


def fused_residadd_rmsnorm(x: Tensor, attn_out: Tensor, scale: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Fused x + scale * attn_out + RMSNorm. Returns (x_new, normed)."""
    return FusedResidAddRMSNorm.apply(x, attn_out, scale, eps)


# -------------------------------------------------------------------------
# Kernel 3: fused SmearGate + RMSNorm
#   gate = sigmoid(raw_gate)
#   x_smeared[b, t, d] = (1 - gate[d]) * x[b,t,d] + gate[d] * x[b,t-1,d]
#   out = rms_norm(x_smeared)
# -------------------------------------------------------------------------

@triton.jit
def _fused_smeargate_rmsnorm_fwd(
    X_ptr, Gate_ptr, Out_ptr,
    stride_batch,   # stride between batch elements (T * D)
    stride_seq,     # stride between sequence positions (D)
    T: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program handles one (batch, seq) row.
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T
    offs = tl.arange(0, D)

    raw_gate = tl.load(Gate_ptr + offs, mask=offs < D).to(tl.float32)
    gate = tl.sigmoid(raw_gate)

    x_cur = tl.load(X_ptr + b * stride_batch + t * stride_seq + offs, mask=offs < D).to(tl.float32)

    if t > 0:
        x_prev = tl.load(X_ptr + b * stride_batch + (t - 1) * stride_seq + offs, mask=offs < D).to(tl.float32)
    else:
        x_prev = tl.zeros([D], dtype=tl.float32)

    y = (1.0 - gate) * x_cur + gate * x_prev

    # RMSNorm
    var = tl.sum(y * y, axis=0) / D
    rrms = 1.0 / tl.sqrt(var + eps)
    out = y * rrms

    tl.store(Out_ptr + b * stride_batch + t * stride_seq + offs, out.to(tl.bfloat16), mask=offs < D)


class FusedSmearGateRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, raw_gate: Tensor, eps: float = 1e-6) -> Tensor:
        assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
        B, T, D = x.shape
        out = torch.empty_like(x)

        _fused_smeargate_rmsnorm_fwd[(B * T,)](
            x, raw_gate.to(x.dtype), out,
            stride_batch=T * D,
            stride_seq=D,
            T=T,
            D=triton.next_power_of_2(D),
            eps=eps,
        )

        ctx.save_for_backward(x, raw_gate)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, raw_gate = ctx.saved_tensors
        B, T, D = x.shape

        # Recompute smeared output in PyTorch for backward
        gate = torch.sigmoid(raw_gate.float())[None, None, :]  # (1, 1, D)
        x_f = x.float()
        prev = torch.nn.functional.pad(x_f[:, :-1, :], (0, 0, 1, 0))
        smeared = (1.0 - gate) * x_f + gate * prev

        # Manual RMSNorm backward (avoids requires_grad_ for torch.compile compat)
        var = (smeared * smeared).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + ctx.eps)
        normed = smeared * rrms
        grad_out_f = grad_out.float()
        grad_smeared = (grad_out_f - normed * (grad_out_f * normed).mean(dim=-1, keepdim=True)) * rrms

        # Backward through smear: smeared = (1-gate)*x + gate*prev
        grad_x = grad_smeared * (1.0 - gate)
        # prev contributions shift back: grad_x[:, t-1] += grad_smeared[:, t] * gate
        grad_x[:, :-1, :] = grad_x[:, :-1, :] + grad_smeared[:, 1:, :] * gate

        # Gradient for raw_gate (through sigmoid)
        sig = torch.sigmoid(raw_gate.float())
        dsig = sig * (1.0 - sig)  # sigmoid derivative
        # d(smeared)/d(gate) = -x + prev, summed over batch and seq
        diff = prev - x_f  # (B, T, D)
        grad_gate = (grad_smeared * diff).sum(dim=(0, 1)) * dsig

        return grad_x.to(x.dtype), grad_gate, None


def fused_smeargate_rmsnorm(x: Tensor, gate: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused SmearGate + RMSNorm. Returns normed output."""
    return FusedSmearGateRMSNorm.apply(x, gate, eps)
