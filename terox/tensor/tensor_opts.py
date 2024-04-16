from typing import Iterable

from terox.autodiff.variable import Variable, VarHistory, VarFunction
from .function import add, sub, mul, div, neg, eq, lt, gt, abs, exp, log, relu

class TensorOptsBackend():
    def __init__(self) -> None:
        self.Add:VarFunction = Add()
        # self.Sub:VarFunction = Sub()
        # self.Mul:VarFunction = Mul()
        # self.Div:VarFunction = Div()
        # self.Neg:VarFunction = Neg()
        # self.Max:VarFunction = Max()
        # self.Min:VarFunction = Min()
        self.Eq:VarFunction = Eq()
        return
    
class Add(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = add(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad, b_grad = grad * grad.new(1.0), grad * grad.new(1.0)
        return a_grad, b_grad
    
class Eq(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = eq(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = grad if a.item() == b.item() else grad.new(0.0)
        b_grad = a_grad
        return a_grad, b_grad