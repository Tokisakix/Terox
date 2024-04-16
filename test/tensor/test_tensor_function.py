import pytest
import numpy as np
from numpy.typing import NDArray

from terox.tensor.function import add, sub, mul, matmul, div, eq, lt, gt, abs, exp, log, relu

@pytest.mark.test_tensor_function
def test_add(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = add(a, b)
    ref = a + b
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_sub(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = sub(a, b)
    ref = a - b
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_mul(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = mul(a, b)
    ref = a * b
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_matmul(a:NDArray=np.array([[1.0, 2.0], [3.0, 4.0]]), b:NDArray=np.array([[2.0, 3.0], [4.0, 5.0]])) -> None:
    res = matmul(a, b)
    ref = a @ b
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_div(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = div(a, b)
    ref = a / b
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_eq(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = eq(a, b)
    ref = np.zeros_like(a)
    ref[a == b] = 1.0
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_lt(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = lt(a, b)
    ref = np.zeros_like(a)
    ref[a < b] = 1.0
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_gt(a:NDArray=np.array([1.0, 2.0, 3.0]), b:NDArray=np.array([2.0, 3.0, 4.0])) -> None:
    res = gt(a, b)
    ref = np.zeros_like(a)
    ref[a > b] = 1.0
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_abs(a:NDArray=np.array([-1.0, -2.0, -3.0])) -> None:
    res = abs(a)
    ref = np.abs(a)
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_exp(a:NDArray=np.array([1.0, 2.0, 3.0])) -> None:
    res = exp(a)
    ref = np.exp(a)
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_log(a:NDArray=np.array([1.0, 2.0, 3.0])) -> None:
    res = log(a)
    ref = np.log(a)
    assert res.all() == ref.all()
    return

@pytest.mark.test_tensor_function
def test_relu(a:NDArray=np.array([-1.0, -2.0, -3.0])) -> None:
    res = relu(a)
    ref = np.copy(a)
    ref[a <= 0.0] = 0.0
    assert res.all() == ref.all()
    return