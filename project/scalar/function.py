from typing import List

from terox.autodiff import Scalar

def softmax(inputs:List[Scalar]) -> List[Scalar]:
    out = [Scalar(0.0) for _ in range(len(inputs))]
    temp = 0.0
    for scalar in inputs:
        temp += scalar.item()
    temp = Scalar(temp)
    for idx in range(len(inputs)):
        out[idx] = inputs[idx] / temp
    return out

def CrossEntropyLoss(inputs:List[Scalar], labels:int) -> Scalar:
    n = len(inputs)
    loss = Scalar(0.0)
    for idx in range(n):
        loss -= (Scalar(1.0) - inputs[idx]).log() if idx == labels else inputs[idx].log()
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