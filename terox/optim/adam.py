import math
from typing import List

from terox.module import Parameter
from .optim import Optimizer

class Adam(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {id(param): 0.0 for param in parameters}
        self.v = {id(param): 0.0 for param in parameters}
        self.t = 0
        return

    def step(self) -> None:
        self.t += 1
        for parameter in self.parameters:
            grad = parameter.value()._gradient._item
            param_id = id(parameter)
            
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            parameter.value()._item -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
        return