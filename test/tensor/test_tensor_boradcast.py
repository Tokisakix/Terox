import pytest
import numpy as np
from numpy.typing import NDArray

from terox.tensor.tensor import Tensor

def TensorEq(a:Tensor, b:Tensor) -> bool:
    return a.item().all() == b.item().all()

@pytest.mark.test_tensor_boradcast
def test_add(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]], b:NDArray=[[1.0, 1.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A + B
    assert TensorEq(Tensor([[0.0, -1.0], [-2.0, -3.0]]), C)
    return

@pytest.mark.test_tensor_boradcast
def test_sub(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]], b:NDArray=[[1.0, 1.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A - B
    assert TensorEq(Tensor([[-2.0, -3.0], [-4.0, -5.0]]), C)
    return

@pytest.mark.test_tensor_boradcast
def test_mul(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]], b:NDArray=[2.0]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A * B
    assert TensorEq(Tensor([[-2.0, -4.0], [-6.0, -8.0]]), C)
    return

@pytest.mark.test_tensor_boradcast
def test_div(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]], b:NDArray=[2.0]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A / B
    assert TensorEq(Tensor([[-0.5, -1.0], [-1.5, -2.0]]), C)
    return