from typing import Iterable, Optional

class VarFunction():
    def __init__(self) -> None:
        return

    def _forward(self, args:Iterable["Variable"]) -> "Variable":
        raise NotImplementedError

    def _backward(self, grad:"Variable", args:Iterable["Variable"]) -> Iterable["Variable"]:
        raise NotImplementedError

class VarHistory():

    _func: VarFunction
    _args: Iterable["Variable"]

    def __init__(self, _func:VarFunction, _args:Iterable["Variable"]) -> None:
        self._func = _func
        self._args = _args
        return

class Variable():

    _history: Optional[VarHistory]
    _gradient: Optional["Variable"]
    _require_grad: bool

    def __init__(self, _history:Optional[VarHistory]=None, _gradient:Optional["Variable"]=None, _require_grad:bool=True) -> None:
        self._history = _history
        self._gradient = _gradient
        self._require_grad = _require_grad
        if self._gradient == None:
            self._zeroGrad()
        pass

    def _parent(self) -> Iterable["Variable"]:
        if self._is_leaf():
            return []
        parent = self._history._args
        return parent

    def _is_leaf(self) -> bool:
        res = self._history == None or not self._require_grad
        return res

    def _chainRule(self) -> None:
        if self._is_leaf():
            return
        args = self._history._args
        func = self._history._func
        grads = func._backward(self._gradient, args)
        for arg, grad in zip(args, grads):
            arg._gradient += grad
        return

    def requireGrad(self, _require_grad:bool) -> None:
        self._require_grad = _require_grad
        if not self._require_grad:
            self._history = None
        return

    def backward(self):
        topoList = getTopoList(self)
        self._gradient = self._oneGrad()
        for variable in topoList:
            variable._chainRule()
        return

    def _zeroGrad(self) -> None:
        raise NotImplementedError

    def _oneGrad(self) -> None:
        raise NotImplementedError

    def detach(self) -> "Variable":
        raise NotImplementedError
    
def _getTopoChain(var:"Variable") -> Iterable["Variable"]:
    topoChain = []
    for parent in var._parent():
        topoChain.append((var, parent))
        topoChain += _getTopoChain(parent)
    return topoChain

def getTopoList(var:"Variable") -> Iterable["Variable"]:
    topoChain = _getTopoChain(var)
    topoChain = list(set(topoChain))
    topoDegree = {var:1}
    for _, parent in topoChain:
        if not parent in topoDegree:
            topoDegree[parent] = 0
        topoDegree[parent] += 1
    topoList, queue = [], [var]
    while len(queue) > 0:
        variable = queue[0]
        queue += variable._parent()
        topoDegree[variable] -= 1
        if topoDegree[variable] == 0:
            topoList.append(variable)
        del queue[0]
    return topoList