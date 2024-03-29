import torch
import torch.nn as nn

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


def gemm_lowbit_kernel(x: torch.Tensor, w: torch.Tensor):
    """
    The standard F.linear operation is replaced with a customized low-bit kernel.
    This is not well-defined in the whitepapers...
    Until I figure out CUDA injection, F.linear is used as a placeholder...
    Consequently, we must use float16 for x and w, for this to work, instead of int8
    """
    y = torch.nn.functional.linear(x, w)
    return y


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-5
        self.quantized_weight = None
        self.weight_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized_weight is None and self.weight_scale is None:
            w = self.weight
            # x_norm = self.norm(x)
            x_norm = rms_norm(x, eps=self.eps)
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            y = nn.functional.linear(x_quant, w_quant)
            return y

        else:
            w = self.quantized_weight
            w_scale = self.weight_scale
            x_quant, x_scale = activation_norm_quant(x)
            y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale
            return y
