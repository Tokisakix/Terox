from random import random

from terox.tensor import Tensor
from terox.module import Module, Parameter

class Linear(Module):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True) -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.pw = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.out_feature)] for _ in range(self.in_feature)]))
        if self.bias:
            self.pb = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.out_feature)]]))
        return
    
    def forward(self, inputs:Tensor) -> Tensor:
        out = inputs @ self.pw.value()
        out = out + self.pb.value()
        return out