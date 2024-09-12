from typing import List

from terox.module import Parameter
from .optim import Optimizer
    
class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float, momentum: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = {id(param): 0.0 for param in parameters}
        return
    
    def step(self) -> None:
        for parameter in self.parameters:
            grad = parameter.value()._gradient._item
            param_id = id(parameter)
            
            self.velocity[param_id] = self.momentum * self.velocity[param_id] - self.lr * grad
            parameter.value()._item += self.velocity[param_id]
        return