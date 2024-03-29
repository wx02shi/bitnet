import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

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
    y = ((x * scale).round().clamp_(-128, 127)).to(torch.int8)
    return y, scale


lowbit_kernel_cuda = """
#include <torch/extension.h>

__global__ void gemm_lowbit_kernel(float* C, const half* A, const char* B, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[i * K + k]) * (float)B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
"""

# Compile the CUDA kernel separately
torch.utils.cpp_extension.load_inline(
    name="gemm_lowbit_kernel_cuda",
    cpp_sources="",
    cuda_sources=lowbit_kernel_cuda,
    extra_cuda_cflags=["--use_fast_math"],
)

lowbit_kernel_cpp = """
#include <torch/extension.h>

void gemm_lowbit(torch::Tensor C, torch::Tensor A, torch::Tensor B) {
    gemm_lowbit_kernel_cuda(
        C.data<float>(),
        A.data<at::Half>(),
        B.data<char>(),
        C.size(0), C.size(1), A.size(1)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_lowbit_kernel", &PyInit_gemm_lowbit_kernel, "GEMM lowbit kernel");
}
"""

# Compile the CUDA kernel
torch.utils.cpp_extension.load_inline(
    name="gemm_lowbit_kernel",
    cpp_sources=lowbit_kernel_cpp,
    extra_cuda_cflags=["--use_fast_math"],
)


def gemm_lowbit(x: torch.Tensor, w: torch.Tensor):
    """
    The standard F.linear operation is replaced with a customized low-bit kernel.
    This is not well-defined in the whitepapers...
    Until I figure out CUDA injection, F.linear is used as a placeholder
    """
    # y = nn.functional.linear(x, w)
    # return y
    assert x.dim() == 2 and w.dim() == 2
    assert x.size(1) == w.size(0)
    M, K = x.size()
    K, N = w.size()

    # Allocate output tensor
    C = torch.zeros(M, N).cuda()

    # Call the CUDA kernel
    gemm_lowbit_kernel.gemm_lowbit(C, x.half(), w.int().to(torch.int8), M, N, K)

    return C


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
            y = gemm_lowbit(x_quant, w).to(torch.float16) / w_scale / x_scale
            return y
