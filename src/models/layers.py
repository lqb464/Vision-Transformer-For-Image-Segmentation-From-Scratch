"""Basic layers for ViT: Linear, LayerNorm, Dropout, Conv2d."""

import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class Dropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x.float()) > self.p).float()
        return mask * x / (1.0 - self.p)


class Conv2d(nn.Module):
    """Custom Conv2d (used for patch embedding)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        N, C_in, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x = torch.nn.functional.pad(x, (pw, pw, ph, ph))
            _, _, H, W = x.shape

        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1

        x_unfold = x.unfold(2, kh, sh).unfold(3, kw, sw)
        x_unfold = x_unfold.contiguous().view(N, C_in * kh * kw, H_out * W_out)

        weight_flat = self.weight.view(self.weight.size(0), -1)
        output = weight_flat @ x_unfold + self.bias.view(1, -1, 1)

        return output.view(N, -1, H_out, W_out)


class ConvTranspose2d(nn.Module):
    """Transposed Convolution (deconvolution) cho upsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in)

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kh, kw).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Dùng PyTorch functional cho transposed conv (quá phức tạp để viết tay hiệu quả)
        return torch.nn.functional.conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )
