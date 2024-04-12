from typing import List

from terox.autodiff import Scalar

def Relu(inputs:List[Scalar]) -> List[Scalar]:
    out = [Scalar(0.0) for _ in range(len(inputs))]
    for idx in range(len(inputs)):
        out[idx] = inputs[idx].relu()
    return out

def Sigmoid(inputs:List[Scalar]) -> List[Scalar]:
    out = [Scalar(0.0) for _ in range(len(inputs))]
    for idx in range(len(inputs)):
        out[idx] = inputs[idx].sigmoid()
    return out

def Softmax(inputs:List[Scalar]) -> List[Scalar]:
    out = [Scalar(0.0) for _ in range(len(inputs))]
    temp = 0.0
    for scalar in inputs:
        temp += scalar.exp().item()
    temp = Scalar(temp)
    for idx in range(len(inputs)):
        out[idx] = inputs[idx].exp() / temp
    return out

def MSELoss(inputs:List[Scalar], labels:List[Scalar]) -> Scalar:
    n = len(inputs)
    loss = Scalar(0.0)
    for idx in range(n):
        temp = inputs[idx] - labels[idx]
        loss += temp * temp / Scalar(2.0)
    loss /= Scalar(n)
    return loss

def argmax(inputs:List[Scalar]) -> int:
    pos = 0
    res = inputs[pos]
    for idx, temp_input in enumerate(inputs):
        if temp_input > res:
            res = temp_input
            pos = idx
    return pos, res