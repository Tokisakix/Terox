from typing import List
from random import random

from terox.tensor import Tensor
from terox.module import Module, Parameter

class ScalarLinear(Module):
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

class ScalarIrisClassifyModel(Module):
    def __init__(self, in_feature, hidden_feature, out_feature) -> None:
        super().__init__()
        self.lr1 = ScalarLinear(in_feature, hidden_feature)
        self.lr2 = ScalarLinear(hidden_feature, out_feature)
        return
    
    def forward(self, inputs:Tensor) -> Tensor:
        out = self.lr1(inputs)
        out = out.relu()
        out = self.lr2(out)
        out = out.sigmoid()
        return out
    
class SGD():
    def __init__(self, parameters:List[Parameter], lr:float) -> None:
        self.parameters = parameters
        self.lr = lr
        return
    
    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.value()._zeroGrad()
        return
    
    def step(self) -> None:
        for parameter in self.parameters:
            # print(parameter.value(), parameter.value()._gradient)
            parameter.value()._item -= self.lr * parameter.value()._gradient._item
        return