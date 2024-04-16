from typing import List

from terox.tensor import Tensor

def MSELoss(inputs:Tensor, labels:Tensor) -> Tensor:
    loss = (inputs - labels) * (inputs - labels)
    return loss