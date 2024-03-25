import torch
from mistral.rms_norm import rms_norm

def activation_quant(x: torch.Tensor, eps: float = 1e-5):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=eps)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w: torch.Tensor, eps: float = 1e-5):
    scale = 1.0 / w.abs().mean().clamp_(min=eps)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

def activation_norm_quant(x: torch.Tensor, eps: float = 1e-5):
    x = rms_norm(x, eps)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=eps)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale

'''
The standard F.linear operation is replaced with a customized low-bit kernel.
This is not well-defined in the whitepapers...
Until I figure out CUDA injection, F.linear is used as a placeholder
'''
def gemm_lowbit_kernel(x: torch.Tensor, w: torch.Tensor):
    y = torch.nn.functional.linear(x, w)
    return y