from typing import Any, Iterable, Optional

class VarFunction():
    def __init__(self) -> None:
        return
    
    def __call__(self, *args: Any, **kwds: Any) -> "Variable":
        return self._forward(*args, **kwds)

    def _forward(self, args:Iterable["Variable"]) -> "Variable":
        raise NotImplementedError

    def _backward(self, grad:"object", args:Iterable["Variable"]) -> Iterable["Variable"]:
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
    _gradient: Optional[object]

    def __init__(self, _history:Optional[VarHistory]=None, _gradient:Optional[object]=None) -> None:
        self._history = _history
        self._gradient = _gradient
        if self._gradient == None:
            self._zeroGrad()
        pass

    def _parent(self) -> Iterable["Variable"]:
        if self._is_leaf():
            return []
        parent = list(self._history._args)
        return parent

    def _is_leaf(self) -> bool:
        res = self._history == None
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

    def backward(self):
        topoList = getTopoList(self)
        self._oneGrad()
        for variable in topoList:
            variable._chainRule()
        return

    def _zeroGrad(self) -> None:
        raise NotImplementedError

    def _oneGrad(self) -> None:
        raise NotImplementedError

    def zero(self) -> "Variable":
        raise NotImplementedError

    def one(self) -> "Variable":
        raise NotImplementedError

    def detach(self) -> "Variable":
        raise NotImplementedError
    
    def item(self) -> object:
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