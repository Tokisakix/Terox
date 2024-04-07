import pytest
import math

from terox.autodiff.scalar import Scalar

@pytest.mark.test_scalar_overload
def test_add_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A + B
    C.backward()
    assert A._gradient == 1.0
    assert B._gradient == 1.0
    return

@pytest.mark.test_scalar_overload
def test_sub(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A - B
    C.backward()
    assert A._gradient == 1.0
    assert B._gradient == -1.0
    return

@pytest.mark.test_scalar_overload
def test_mul(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A * B
    C.backward()
    assert A._gradient == 3.0
    assert B._gradient == 2.0
    return

@pytest.mark.test_scalar_overload
def test_div(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A / B
    C.backward()
    assert A._gradient == 1.0 / 3.0
    assert B._gradient == -1.0 * 2.0 / 9.0
    return

@pytest.mark.test_scalar_overload
def test_neg(a:float=2.0) -> None:
    A = Scalar(a, None, None)
    C = -A
    C.backward()
    assert A._gradient == -1.0
    return

@pytest.mark.test_scalar_overload
def test_max(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A.max(A, B)
    C.backward()
    assert A._gradient == 0.0
    assert B._gradient == 1.0
    return

@pytest.mark.test_scalar_overload
def test_min(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A.min(A, B)
    C.backward()
    assert A._gradient == 1.0
    assert B._gradient == 0.0
    return

@pytest.mark.test_scalar_overload
def test_eq(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A == B
    C.backward()
    assert A._gradient == 0.0
    assert B._gradient == 0.0
    return

@pytest.mark.test_scalar_overload
def test_lt(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A < B
    C.backward()
    assert A._gradient == 0.0
    assert B._gradient == 0.0
    return

@pytest.mark.test_scalar_overload
def test_gt(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a, None, None)
    B = Scalar(b, None, None)
    C = A > B
    C.backward()
    assert A._gradient == 0.0
    assert B._gradient == 0.0
    return

@pytest.mark.test_scalar_overload
def test_abs(a:float=-2.0) -> None:
    A = Scalar(a, None, None)
    C = A.abs(A)
    C.backward()
    assert A._gradient == -1.0
    return

@pytest.mark.test_scalar_overload
def test_exp(a:float=2.0) -> None:
    A = Scalar(a, None, None)
    C = A.exp()
    C.backward()
    assert A._gradient == math.exp(2.0)
    return

@pytest.mark.test_scalar_overload
def test_log(a:float=2.0) -> None:
    A = Scalar(a, None, None)
    C = A.log()
    C.backward()
    assert A._gradient == 1.0 / 2.0
    return

@pytest.mark.test_scalar_overload
def test_relu(a:float=-2.0) -> None:
    A = Scalar(a, None, None)
    C = A.relu()
    C.backward()
    assert A._gradient == 0.0
    return

@pytest.mark.test_scalar_overload
def test_sigmoid(a:float=-2.0) -> None:
    A = Scalar(a, None, None)
    C = A.sigmoid()
    C.backward()
    sigmoid = 1.0 / (1.0 + math.exp(2.0))
    assert A._gradient == sigmoid * (1.0 - sigmoid)
    return

@pytest.mark.test_scalar_overload
def test_tanh(a:float=-2.0) -> None:
    A = Scalar(a, None, None)
    C = A.tanh()
    C.backward()
    tanh = (math.exp(-2.0) - math.exp(2.0)) / (math.exp(-2.0) + math.exp(2.0))
    assert A._gradient == 1.0 - tanh ** 2
    return

@pytest.mark.test_scalar_overload
def test_complex_backward() -> None:
    A = Scalar(1.0, None, None)
    B = Scalar(2.0, None, None)
    C = A + B
    D = A + C
    E = (D - C) * (D - C)
    E.backward()
    assert A._gradient == 2.0
    assert B._gradient == 0.0
    assert C._gradient == 0.0
    assert D._gradient == 2.0
    return