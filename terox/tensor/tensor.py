import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Iterable

from ..autodiff.variable import Variable, VarHistory
from .tensor_opts import TensorOptsBackend

tensor_count:int = 0

class Tensor(Variable):

    _id: int
    _item: NDArray

    def __init__(self, _item:Iterable=None, _history:Optional[VarHistory]=None, _gradient:Optional["Tensor"]=None, _require_grad:bool=True, _backend:TensorOptsBackend=TensorOptsBackend()) -> None:
        super().__init__(_history, _gradient, _require_grad)
        global tensor_count
        self._id = tensor_count
        tensor_count += 1
        if _item is None:
            _item = [0.0]
        self._item = np.array(list(_item))
        self._backend = _backend
        return

    def _zeroGrad(self) -> None:
        self._gradient = Tensor(np.zeros_like(self._item), _require_grad=False)
        return

    def _oneGrad(self) -> None:
        self._gradient = Tensor(np.ones_like(self._item), _require_grad=False)
        return
    
    def new(self, _item:Iterable=None, _history:Optional[VarHistory]=None, _gradient:Optional["Tensor"]=None, _require_grad:bool=True) -> "Tensor":
        if _item is None:
            _item = [0.0]
        _item = np.array(list(_item))
        res = Tensor(_item, _history, _gradient, _require_grad)
        return res

    def zero(self, _shape:Iterable) -> "Tensor":
        var = Tensor(np.zeros(_shape))
        return var

    def one(self, _shape:Iterable) -> "Tensor":
        var = Tensor(np.ones(_shape))
        return var

    def detach(self) -> "Tensor":
        var = Tensor(
            _item=self._item,
            _history=None,
            _gradient=None,
            _require_grad=self._require_grad,
        )
        return var

    def item(self) -> Iterable:
        return self._item
    
    def shape(self) -> Tuple:
        return self._item.shape
    
    def reshape(self, _shape:Iterable) -> "Tensor":
        self._item = self._item.reshape(_shape)
        return self
    
    def __str__(self) -> str:
        info = f"<{self.__class__.__name__}({self._item}), grad_fn="
        info += f"None" if self._history == None else f"{self._history._func.__class__.__name__}"
        info += f", grad={self._require_grad}"
        info += ">"
        return info
    
    def __getitem__(self, key:Iterable) -> NDArray:
        return self._item[key]

    def __setitem__(self, key:Iterable, val: Iterable) -> None:
        self._item[key] = val
        return
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __add__(self, b:"Tensor") -> "Tensor":
        return self._backend.Add(self, b)
    
    def __sub__(self, b:"Tensor") -> "Tensor":
        return self._backend.Sub(self, b)
    
    def __mul__(self, b:"Tensor") -> "Tensor":
        return self._backend.Mul(self, b)
    
    def __truediv__(self, b:"Tensor") -> "Tensor":
        return self._backend.Div(self, b)
    
    def __neg__(self) -> "Tensor":
        return self._backend.Neg(self)
    
    def abs(self, a:"Tensor") -> "Tensor":
        return self._backend.Abs(a)
    
    def __eq__(self, b:"Tensor") -> "Tensor":
        return self._backend.Eq(self, b)
    
    def __lt__(self, b:"Tensor") -> "Tensor":
        return self._backend.Lt(self, b)
    
    def __gt__(self, b:"Tensor") -> "Tensor":
        return self._backend.Gt(self, b)
    
    def exp(self) -> "Tensor":
        return self._backend.Exp(self)
    
    def log(self) -> "Tensor":
        return self._backend.Log(self)
    
    def relu(self) -> "Tensor":
        return self._backend.Relu(self)
    
    def sigmoid(self) -> "Tensor":
        return self._backend.Sigmoid(self)
    
    def tanh(self) -> "Tensor":
        return self._backend.Tanh(self)