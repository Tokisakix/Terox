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

    _id:int
    _item:object
    _history: Optional[VarHistory]
    _gradient: Optional["Variable"]
    _require_grad: bool

    def __init__(self, _history:Optional[VarHistory]=None, _gradient:Optional["Variable"]=None, _require_grad:bool=True) -> None:
        self._history = _history
        self._gradient = _gradient
        self._require_grad = _require_grad
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
        TopoList = getTopoList(self)
        self._oneGrad()
        for var in TopoList:
            var._chainRule()
        return
    
    def setRequireGrad(self, _require_grad:bool=True) -> "Variable":
        self._require_grad = _require_grad
        return self
    
    def getRequireGrad(self) -> bool:
        return self._require_grad
    
    def new(self) -> "Variable":
        raise NotImplementedError

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
    
    def __str__(self) -> str:
        raise NotImplementedError
    
def _getTopoChain(var:"Variable") -> Iterable["Variable"]:
    topoChainId = []
    topoChainVar = []
    for parent in var._parent():
        topoChainId.append((var._id, parent._id))
        topoChainVar.append(parent)
        temp = _getTopoChain(parent)
        topoChainId += temp[0]
        topoChainVar += temp[1]
    return topoChainId, topoChainVar

def getTopoList(var:"Variable") -> Iterable["Variable"]:
    topoChainId, topoChainVar = _getTopoChain(var)
    topoId2Var = {var._id:var}
    topoId2Degree = {var._id:1}
    for (_, temp_id), temp_var in zip(topoChainId, topoChainVar):
        topoId2Var[temp_id] = temp_var
    topoChainId = list(set(topoChainId))
    for _, parent_id in topoChainId:
        if not parent_id in topoId2Degree:
            topoId2Degree[parent_id] = 0
        topoId2Degree[parent_id] += 1
    topoList, queue = [], [var]
    while len(queue) > 0:
        variable = queue[0]
        queue += variable._parent()
        topoId2Degree[variable._id] -= 1
        if topoId2Degree[variable._id] == 0:
            variable._zeroGrad()
            topoList.append(variable)
        del queue[0]
    return topoList