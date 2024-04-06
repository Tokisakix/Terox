from typing import Iterable

from terox.autodiff.variable import Variable

from .variable import VarFunction, VarHistory, Variable
from .function import add, sub, mul, div, neg, max, min, eq, lt, gt, abs, exp, log, relu

class ScalarOptsBackend():
    def __init__(self) -> None:
        self.Add:VarFunction = Add()
        self.Sub:VarFunction = Sub()
        self.Mul:VarFunction = Mul()
        self.Div:VarFunction = Div()
        self.Neg:VarFunction = Neg()
        self.Max:VarFunction = Max()
        self.Min:VarFunction = Min()
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
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        a_grad, b_grad = grad * 1.0, grad * 1.0
        return a_grad, b_grad

class Sub(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = sub(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        a_grad, b_grad = grad * 1.0, grad * -1.0
        return a_grad, b_grad

class Mul(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = mul(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad, b_grad = grad * b.item(), grad * a.item()
        return a_grad, b_grad

class Div(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = div(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = grad / b.item()
        b_grad = -grad * a.item() / (b.item() ** 2)
        return a_grad, b_grad

class Neg(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = neg(a.item())
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = -grad
        return (a_grad,)

class Max(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = max(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history,  None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = grad if a.item() >= b.item() else 0.0
        b_grad = grad if b.item() >= a.item() else 0.0
        return a_grad, b_grad

class Min(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = min(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = grad if a.item() <= b.item() else 0.0
        b_grad = grad if b.item() <= a.item() else 0.0
        return a_grad, b_grad

class Eq(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = eq(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = grad if a.item() == b.item() else 0.0
        b_grad = a_grad
        return a_grad, b_grad

class Lt(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = lt(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = 0.0
        b_grad = 0.0
        return a_grad, b_grad

class Gt(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, b:Variable) -> Variable:
        _item = gt(a.item(), b.item())
        _history = VarHistory(self, (a, b))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = 0.0
        b_grad = 0.0
        return a_grad, b_grad

class Abs(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = abs(a.item())
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad if a.item() >= 0.0 else -grad
        return (a_grad,)

class Exp(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = exp(a.item())
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad * exp(a.item())
        return (a_grad,)

class Log(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = log(a.item())
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad / a.item()
        return (a_grad,)

class Relu(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = relu(a.item())
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad if a.item() >= 0.0 else 0.0
        return (a_grad,)

class Sigmoid(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = 1.0 / (1.0 + exp(-a.item()))
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        sigmoid = 1.0 / (1.0 + exp(-a.item()))
        a_grad = grad * sigmoid * (1.0 - sigmoid)
        return (a_grad,)

class Tanh(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = (exp(a.item()) - exp(-a.item())) / (exp(a.item()) + exp(-a.item()))
        _history = VarHistory(self, (a,))
        res = a.new(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        tanh = (exp(a.item()) - exp(-a.item())) / (exp(a.item()) + exp(-a.item()))
        a_grad = grad * (1.0 - tanh ** 2)
        return (a_grad,)