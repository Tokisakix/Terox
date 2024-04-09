from typing import List
from random import random

from terox.autodiff import Scalar
from terox.module import Module, Parameter

class ScalarLinear(Module):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True) -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        for i in range(in_feature):
            for j in range(out_feature):
                self.__dict__[f"w{i}{j}"] = Parameter(Scalar(random() - 1))
        if self.bias:
            for j in range(out_feature):
                self.__dict__[f"b{j}"] = Parameter(Scalar(random() - 1))
        return
    
    def forward(self, inputs:List[Scalar]) -> List[Scalar]:
        out = [Scalar(0.0) for _ in range(self.out_feature)]
        for i in range(self.in_feature):
            for j in range(self.out_feature):
                out[j] += inputs[i] * self.__dict__[f"w{i}{j}"].value()
        if self.bias:
            for j in range(self.out_feature):
                out[j] += self.__dict__[f"b{j}"].value()
        return out

class ScalarIrisClassifyModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.lr1 = ScalarLinear(in_feature=4, out_feature=32)
        self.lr2 = ScalarLinear(in_feature=32, out_feature=3)
        return
    
    def forward(self, inputs:List[Scalar]) -> List[Scalar]:
        out:List[Scalar] = self.lr1(inputs)
        for idx in range(len(out)):
            out[idx] = out[idx].relu()
        out:List[Scalar] = self.lr2(out)
        return out
    
class GD():
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
            parameter.value()._item -= self.lr * parameter.value()._gradient._item
        return