import pytest
import numpy as np

from terox.autodiff.variable import VarHistory, VarFunction
from terox.tensor.tensor import Tensor

@pytest.mark.test_tensor
def test_tensor_init() -> None:
    a = Tensor([1.0])
    b = Tensor([2.0])
    Add = VarFunction()
    history = VarHistory(Add, (a, b))
    gradient = Tensor([1.0], _require_grad=False)
    c = Tensor([3.0], history, gradient)
    assert c._parent() == [a, b]
    assert not c._is_leaf()
    assert c._gradient == gradient
    return

@pytest.mark.test_tensor
def test_tensor_zero_grad() -> None:
    a = Tensor([0.0, 0.0], None, [1.0, 1.0])
    a._zeroGrad()
    assert a._gradient == Tensor([0.0, 0.0])
    return

@pytest.mark.test_tensor
def test_tensor_one_grad() -> None:
    a = Tensor([0.0, 0.0], None, [0.0, 0.0])
    a._oneGrad()
    assert a._gradient == Tensor([1.0, 1.0])
    return

@pytest.mark.test_tensor
def test_tensor_zero() -> None:
    a = Tensor().zero([2, 1])
    assert a._item.all() == Tensor([[0.0], [0.0]])._item.all()
    return

@pytest.mark.test_tensor
def test_tensor_one() -> None:
    a = Tensor().one([2, 1])
    assert a._item.all() == Tensor([[1.0], [1.0]])._item.all()
    return

@pytest.mark.test_tensor
def test_tensor_item() -> None:
    a = Tensor([1.0])
    assert a.item() == [1.0]
    return

@pytest.mark.test_tensor
def test_tensor_str() -> None:
    a = Tensor([1.5])
    b = Tensor([2.0])
    c = a + b
    assert str(a) == "<Tensor([1.5]), grad_fn=None, grad=True>"
    assert str(c) == "<Tensor([3.5]), grad_fn=Add, grad=True>"
    return

@pytest.mark.test_tensor
def test_tensor_reshape() -> None:
    a = Tensor().zero([3, 2, 1])
    assert a.shape() == (3, 2, 1)
    a = a.reshape([6, 1])
    assert a.shape() == (6, 1)
    return