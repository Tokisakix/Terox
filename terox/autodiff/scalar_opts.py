from typing import Iterable

from terox.autodiff.variable import Variable

from .variable import VarFunction, VarHistory
from .scalar import Scalar
from .function import add, sub, mul, div, inv, neg, max, min, eq, lt, gt, abs, exp, log, relu

class ScalarOptsBackend():
    def __init__(self) -> None:
        self.Add:VarFunction = Add()
        self.Sub:VarFunction = Sub()
        self.Mul:VarFunction = Mul()
        self.Div:VarFunction = Div()
        self.Inv:VarFunction = Inv()
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
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = add(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        a_grad, b_grad = grad * 1.0, grad * 1.0
        return a_grad, b_grad

class Sub(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = sub(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        a_grad, b_grad = grad * 1.0, grad * -1.0
        return a_grad, b_grad

class Mul(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = mul(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad, b_grad = grad * b.item(), grad * a.item()
        return a_grad, b_grad

class Div(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = div(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a, b) = args
        a_grad = grad / b.item()
        b_grad = -grad * a.item() / (b.item() ** 2)
        return a_grad, b_grad

class Inv(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = inv(a._item)
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = -grad / (a.item() ** 2)
        return a_grad

class Neg(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = neg(a._item)
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = -grad
        return a_grad

class Max(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = max(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history,  None)
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
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = min(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
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
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = eq(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
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
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = lt(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
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
    
    def _forward(self, a:Scalar, b:Scalar) -> Scalar:
        _item = gt(a._item, b._item)
        _history = VarHistory(self, (a, b))
        res = Scalar(_item, _history, None)
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
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = abs(a._item)
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad if a.item() >= 0.0 else -grad
        return a_grad

class Exp(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = exp(a._item)
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad * exp(a.item())
        return a_grad

class Log(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = log(a._item)
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad / a.item()
        return a_grad

class Relu(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = relu(a._item)
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad if a.item() >= 0.0 else 0.0
        return a_grad

class Sigmoid(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = 1.0 / (1.0 + exp(-a._item))
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        sigmoid = 1.0 / (1.0 + exp(-a.item()))
        a_grad = grad * sigmoid * (1.0 - sigmoid)
        return a_grad

class Tanh(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Scalar) -> Scalar:
        _item = (exp(a._item) - exp(-a._item)) / (exp(a._item) + exp(-a._item))
        _history = VarHistory(self, (a,))
        res = Scalar(_item, _history, None)
        return res
    
    def _backward(self, grad:float, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        tanh = (exp(a.item()) - exp(-a.item())) / (exp(a.item()) + exp(-a.item()))
        a_grad = grad * (1.0 - tanh ** 2)
        return a_grad