from random import random

from terox.tensor import Tensor
from terox.module import Module, Parameter

class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.pw = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.kernel_size)] for _ in range(self.in_channels)]))
        if self.bias:
            self.pb = Parameter(Tensor([2 * (random() - 0.5) for _ in range(self.out_channels)]))
        return

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs._backend.Conv1d(inputs, self.pw.value(), self.stride, self.padding)
        if self.bias:
            out = out + self.pb.value()
        return out

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.pw = Parameter(Tensor([[[2 * (random() - 0.5) for _ in range(self.kernel_size[1])] for _ in range(self.kernel_size[0])] for _ in range(self.in_channels)]))
        if self.bias:
            self.pb = Parameter(Tensor([2 * (random() - 0.5) for _ in range(self.out_channels)]))
        return

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs._backend.Conv2d(inputs, self.pw.value(), self.stride, self.padding)
        if self.bias:
            out = out + self.pb.value()
        return out

class Conv3d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.pw = Parameter(Tensor([[[[2 * (random() - 0.5) for _ in range(self.kernel_size[2])] for _ in range(self.kernel_size[1])] for _ in range(self.kernel_size[0])] for _ in range(self.in_channels)]))
        if self.bias:
            self.pb = Parameter(Tensor([2 * (random() - 0.5) for _ in range(self.out_channels)]))
        return

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs._backend.Conv3d(inputs, self.pw.value(), self.stride, self.padding)
        if self.bias:
            out = out + self.pb.value()
        return out