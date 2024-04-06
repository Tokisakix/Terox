import pytest

from terox.autodiff.variable import VarHistory, VarFunction
from terox.autodiff.scalar import Scalar

@pytest.mark.test_scalar
def test_scalar_init() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    Add = VarFunction()
    history = VarHistory(Add, (a, b))
    gradient = 1.0
    c = Scalar(3.0, history, gradient)
    assert c._parent() == [a, b]
    assert not c._is_leaf()
    assert c._gradient == gradient
    return

@pytest.mark.test_scalar
def test_scalar_zero_grad() -> None:
    a = Scalar(0.0, None, 1.0)
    a._zeroGrad()
    assert a._gradient == 0.0
    return

@pytest.mark.test_scalar
def test_scalar_one_grad() -> None:
    a = Scalar(0.0, None, 0.0)
    a._oneGrad()
    assert a._gradient == 1.0
    return

@pytest.mark.test_scalar
def test_scalar_zero() -> None:
    a = Scalar().zero()
    assert a._item == 0.0
    return

@pytest.mark.test_scalar
def test_scalar_one() -> None:
    a = Scalar().one()
    assert a._item == 1.0
    return

@pytest.mark.test_scalar
def test_scalar_item() -> None:
    a = Scalar(1.0)
    assert a.item() == 1.0
    return