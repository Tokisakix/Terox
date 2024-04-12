from typing import List
from random import random

from terox.autodiff import Scalar
from terox.module import Module, Parameter
from function import Relu, Sigmoid, Softmax

class ScalarLinear(Module):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True) -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        for i in range(in_feature):
            for j in range(out_feature):
                self.__dict__[f"w{i}{j}"] = Parameter(Scalar(random() - 0.5))
        if self.bias:
            for j in range(out_feature):
                self.__dict__[f"b{j}"] = Parameter(Scalar(random() - 0.5))
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
    def __init__(self, in_feature, hidden_feature, out_feature) -> None:
        super().__init__()
        self.lr1 = ScalarLinear(in_feature, hidden_feature)
        self.lr2 = ScalarLinear(hidden_feature, out_feature)
        return
    
    def forward(self, inputs:List[Scalar]) -> List[Scalar]:
        out = self.lr1(inputs)
        out = Sigmoid(out)
        out = self.lr2(out)
        out = Softmax(out)
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
            # print(parameter.value(), parameter.value()._gradient)
            parameter.value()._item -= self.lr * parameter.value()._gradient._item
        return