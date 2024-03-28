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


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-5
        # self.norm = RMSNorm(out_features, eps=self.eps)
        self.quantized_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # x_norm = self.norm(x)
        x_norm = rms_norm(x, eps=self.eps)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = nn.functional.linear(x_quant, w_quant)
        return y
