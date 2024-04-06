from typing import Optional

from .variable import Variable, VarHistory
from .scalar_opts import ScalarOptsBackend

class Scalar(Variable):

    _item: float

    def __init__(self, _item:float=None, _history:Optional[VarHistory]=None, _gradient:Optional["Scalar"]=None, _backend:ScalarOptsBackend=ScalarOptsBackend()) -> None:
        super().__init__(_history, _gradient)
        if _item == None:
            _item = 0.0
        self._item = float(_item)
        self._backend = _backend
        return

    def _zeroGrad(self) -> None:
        self._gradient = 0.0
        return

    def _oneGrad(self) -> None:
        self._gradient = 1.0
        return
    
    def new(self, _item:float=None, _history:Optional[VarHistory]=None, _gradient:Optional["Scalar"]=None) -> "Scalar":
        if _item == None:
            _item = 0.0
        _item = float(_item)
        res = Scalar(_item, _history, _gradient)
        return res

    def zero(self) -> "Scalar":
        var = Scalar(0.0)
        return var

    def one(self) -> "Scalar":
        var = Scalar(1.0)
        return var

    def detach(self) -> "Scalar":
        var = Scalar(
            _item=self._item,
            _history=None,
            _gradient=None,
        )
        return var

    def item(self) -> float:
        return self._item
    
    def __add__(self, b:"Scalar") -> "Scalar":
        return self._backend.Add(self, b)
    
    def __sub__(self, b:"Scalar") -> "Scalar":
        return self._backend.Sub(self, b)
    
    def __mul__(self, b:"Scalar") -> "Scalar":
        return self._backend.Mul(self, b)
    
    def __truediv__(self, b:"Scalar") -> "Scalar":
        return self._backend.Div(self, b)
    
    def __neg__(self) -> "Scalar":
        return self._backend.Neg(self)
    
    def max(self, a:"Scalar", b:"Scalar") -> "Scalar":
        return self._backend.Max(a, b)
    
    def min(self, a:"Scalar", b:"Scalar") -> "Scalar":
        return self._backend.Min(a, b)
    
    def abs(self, a:"Scalar") -> "Scalar":
        return self._backend.Abs(a)
    
    def __eq__(self, b:"Scalar") -> "Scalar":
        return self._backend.Eq(self, b)
    
    def __lt__(self, b:"Scalar") -> "Scalar":
        return self._backend.Lt(self, b)
    
    def __gt__(self, b:"Scalar") -> "Scalar":
        return self._backend.Gt(self, b)
    
    def exp(self) -> "Scalar":
        return self._backend.Exp(self)
    
    def log(self) -> "Scalar":
        return self._backend.Log(self)
    
    def relu(self) -> "Scalar":
        return self._backend.Relu(self)
    
    def sigmoid(self) -> "Scalar":
        return self._backend.Sigmoid(self)
    
    def tanh(self) -> "Scalar":
        return self._backend.Tanh(self)