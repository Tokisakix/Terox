from typing import Iterable

from terox.autodiff.variable import Variable, VarHistory, VarFunction
from .function import add, sub, mul, matmul, div, neg, eq, lt, gt, abs, exp, log, relu

class TensorOptsBackend():
    def __init__(self) -> None:
        self.Add:VarFunction = Add()
        self.Sub:VarFunction = Sub()
        self.Mul:VarFunction = Mul()
        self.Matmul:VarFunction = Matmul()
        self.Div:VarFunction = Div()
        self.Neg:VarFunction = Neg()
        self.Eq:VarFunction = Eq()
        self.Lt:VarFunction = Lt()
        self.Gt:VarFunction = Gt()
        self.Abs:VarFunction = Abs()
        self.Exp:VarFunction = Exp()
        self.Log:VarFunction = Log()
        self.Relu:VarFunction = Relu()
        self.Sigmoid:VarFunction = Sigmoid()
        self.Tanh:VarFunction = Tanh()
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
        pass
    
class Sub(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = sub(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Mul(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = mul(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Matmul(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = matmul(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Div(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = div(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Neg(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = neg(a.item())
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
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
        pass
    
class Lt(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = lt(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Gt(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = gt(a.item(), b.item())
        _require_grad = a.getRequireGrad() and b.getRequireGrad()
        _history = VarHistory(self, (a, b)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Abs(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = abs(a.item())
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Exp(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = exp(a.item())
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Log(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = log(a.item())
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Relu(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = relu(a.item())
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Sigmoid(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = 1.0 / (1.0 + exp(-a.item()))
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass
    
class Tanh(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = (exp(a.item()) - exp(-a.item())) / (exp(a.item()) + exp(-a.item()))
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        pass