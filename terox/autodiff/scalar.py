from typing import Optional

from .variable import Variable, VarHistory

class Scalar(Variable):

    _item: float

    def __init__(self, _item:float=None, _history:Optional[VarHistory]=None, _gradient:Optional["Scalar"]=None) -> None:
        super().__init__(_history, _gradient)
        if _item == None:
            _item = 0.0
        self._item = float(_item)
        return

    def _zeroGrad(self) -> None:
        self._gradient = 0.0
        return

    def _oneGrad(self) -> None:
        self._gradient = 1.0
        return

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