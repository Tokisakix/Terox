from typing import List

from terox.module import Parameter

class Optimizer():
    def __init__(self, parameters: List[Parameter], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr
        return

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.value()._zeroGrad()
        return
    
    def step(self) -> None:
        raise NotImplementedError