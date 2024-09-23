from typing import List

from terox.tensor import Tensor

def MSELoss(inputs:Tensor, labels:Tensor) -> Tensor:
    loss = (inputs - labels) * (inputs - labels)
    loss = loss.mean()
    return loss

def MAELoss(inputs: Tensor, labels: Tensor) -> Tensor:
    loss = (inputs - labels).abs()
    loss = loss.mean()
    return loss

def CrossEntropyLoss(inputs: Tensor, labels: Tensor) -> Tensor:
    loss = - (labels * inputs.softmax(dim=-1).log()).sum(dim=-1)
    loss = loss.mean()
    return loss

