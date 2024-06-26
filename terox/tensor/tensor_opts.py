from typing import Iterable

from terox.autodiff.variable import Variable, VarHistory, VarFunction
from .function import add, sub, mul, matmul, tranpose, div, neg, eq, lt, gt, abs, exp, log, relu, permute, reshape

class TensorOptsBackend():
    def __init__(self) -> None:
        self.Add:VarFunction = Add()
        self.Sub:VarFunction = Sub()
        self.Mul:VarFunction = Mul()
        self.Tranpose:VarFunction = Tranpose()
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
        self.Reshape:VarFunction = Reshape()
        self.Permute:VarFunction = Permute()
        self.Sum:VarFunction = Sum()
        self.Mean:VarFunction = Mean()
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
        a_grad = grad * grad.one(a.shape())
        b_grad = grad * grad.one(b.shape())
        return a_grad, b_grad
    
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
        (a, b) = args
        a_grad = grad * grad.one(a.shape())
        b_grad = grad * -grad.one(b.shape())
        return a_grad, b_grad
    
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
        (a, b) = args
        a_grad = grad * b
        b_grad = grad * a
        return a_grad, b_grad
    
class Tranpose(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable) -> Variable:
        _item = tranpose(a.item())
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad.tranpose()
        return a_grad
    
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
        (a, b) = args
        a_grad = grad @ b.tranpose()
        b_grad = a.tranpose() @ grad
        return a_grad, b_grad
    
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
        (a, b) = args
        a_grad = grad / b
        b_grad = -grad / (b * b)
        return a_grad, b_grad
    
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
        (a,) = args
        a_grad = -grad
        return a_grad,
    
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
        grad._item[a._item != b._item] = 0.0
        a_grad = grad
        b_grad = grad
        return a_grad, b_grad
    
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
        (a, b) = args
        a_grad = grad.zero(a.shape())
        b_grad = grad.zero(b.shape())
        return a_grad, b_grad
    
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
        (a, b) = args
        a_grad = grad.zero(a.shape())
        b_grad = grad.zero(b.shape())
        return a_grad, b_grad
    
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
        (a,) = args
        grad[a._item < 0.0] = -grad[a._item < 0.0]
        a_grad = grad
        return a_grad,
    
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
        (a,) = args
        a_grad = grad * a.exp()
        return a_grad,
    
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
        (a,) = args
        a_grad = grad / a
        return a_grad,
    
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
        (a,) = args
        grad[a._item <= 0.0] = 0.0
        a_grad = grad
        return a_grad,
    
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
        (a,) = args
        sigmoid = 1.0 / (1.0 + exp(a.item()))
        a_grad = grad * grad.new(sigmoid * (1.0 - sigmoid))
        return a_grad,
    
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
        (a,) = args
        tanh = (exp(a.item())) - exp(-a.item()) / (exp(a.item()) + exp(-a.item()))
        a_grad = grad * grad.new(1.0 - tanh * tanh)
        return a_grad,
    
class Reshape(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, _shape:Iterable) -> Variable:
        _item = reshape(a._item, _shape)
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad.reshape(a.shape())
        return a_grad,
    
class Permute(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, order:Iterable) -> Variable:
        self.order = [order[i] for i in order]
        _item = permute(a._item, order)
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad.permute(self.order)
        return a_grad,
    
class Sum(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, dim:int) -> Variable:
        _item = a._item.sum(dim)
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad.zero(a.shape()) + grad
        return a_grad,
    
class Mean(VarFunction):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def _forward(self, a:Variable, dim:int) -> Variable:
        _item = a._item.mean(dim)
        _require_grad = a.getRequireGrad()
        _history = VarHistory(self, (a,)) if _require_grad else None
        res = a.new(_item, _history, None, _require_grad)
        return res
    
    def _backward(self, grad:Variable, args: Iterable[Variable]) -> Iterable[Variable]:
        (a,) = args
        a_grad = grad.zero(a.shape()) + grad
        return a_grad,