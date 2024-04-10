import pytest
import math

from terox.autodiff.scalar import Scalar

@pytest.mark.test_scalar_backward
def test_add_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A + B
    C.backward()
    assert A._gradient == Scalar(1.0)
    assert B._gradient == Scalar(1.0)
    return

@pytest.mark.test_scalar_backward
def test_sub_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A - B
    C.backward()
    assert A._gradient == Scalar(1.0)
    assert B._gradient == Scalar(-1.0)
    return

@pytest.mark.test_scalar_backward
def test_mul_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A * B
    C.backward()
    assert A._gradient == Scalar(3.0)
    assert B._gradient == Scalar(2.0)
    return

@pytest.mark.test_scalar_backward
def test_div_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A / B
    C.backward()
    assert A._gradient == Scalar(1.0 / 3.0)
    assert B._gradient == Scalar(-1.0 * 2.0 / 9.0)
    return

@pytest.mark.test_scalar_backward
def test_neg_backward(a:float=2.0) -> None:
    A = Scalar(a)
    C = -A
    C.backward()
    assert A._gradient == Scalar(-1.0)
    return

@pytest.mark.test_scalar_backward
def test_max_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A.max(A, B)
    C.backward()
    assert A._gradient == Scalar(0.0)
    assert B._gradient == Scalar(1.0)
    return

@pytest.mark.test_scalar_backward
def test_min_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A.min(A, B)
    C.backward()
    assert A._gradient == Scalar(1.0)
    assert B._gradient == Scalar(0.0)
    return

@pytest.mark.test_scalar_backward
def test_eq_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A == B
    C.backward()
    assert A._gradient == Scalar(0.0)
    assert B._gradient == Scalar(0.0)
    return

@pytest.mark.test_scalar_backward
def test_lt_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A < B
    C.backward()
    assert A._gradient == Scalar(0.0)
    assert B._gradient == Scalar(0.0)
    return

@pytest.mark.test_scalar_backward
def test_gt_backward(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = A > B
    C.backward()
    assert A._gradient == Scalar(0.0)
    assert B._gradient == Scalar(0.0)
    return

@pytest.mark.test_scalar_backward
def test_abs_backward(a:float=-2.0) -> None:
    A = Scalar(a)
    C = A.abs(A)
    C.backward()
    assert A._gradient == Scalar(-1.0)
    return

@pytest.mark.test_scalar_backward
def test_exp_backward(a:float=2.0) -> None:
    A = Scalar(a)
    C = A.exp()
    C.backward()
    assert A._gradient == Scalar(math.exp(2.0))
    return

@pytest.mark.test_scalar_backward
def test_log_backward(a:float=2.0) -> None:
    A = Scalar(a)
    C = A.log()
    C.backward()
    assert A._gradient == Scalar(1.0 / 2.0)
    return

@pytest.mark.test_scalar_backward
def test_relu_backward(a:float=-2.0) -> None:
    A = Scalar(a)
    C = A.relu()
    C.backward()
    assert A._gradient == Scalar(0.0)
    return

@pytest.mark.test_scalar_backward
def test_sigmoid_backward(a:float=-2.0) -> None:
    A = Scalar(a)
    C = A.sigmoid()
    C.backward()
    sigmoid = 1.0 / (1.0 + math.exp(2.0))
    assert A._gradient == Scalar(sigmoid * (1.0 - sigmoid))
    return

@pytest.mark.test_scalar_backward
def test_tanh_backward(a:float=-2.0) -> None:
    A = Scalar(a)
    C = A.tanh()
    C.backward()
    tanh = (math.exp(-2.0) - math.exp(2.0)) / (math.exp(-2.0) + math.exp(2.0))
    assert A._gradient == Scalar(1.0 - tanh ** 2)
    return

@pytest.mark.test_scalar_backward
def test_complex_backward() -> None:
    A = Scalar(1.0)
    B = Scalar(2.0)
    C = A + B
    D = A + C
    E = (D - C) * (D - C)
    E.backward()
    assert A._gradient == Scalar(2.0)
    assert B._gradient == Scalar(0.0)
    assert C._gradient == Scalar(0.0)
    assert D._gradient == Scalar(2.0)
    return

@pytest.mark.test_scalar_backward
def test_scalar_str() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    c.backward()
    assert str(a._gradient) == "<Scalar(1.0), grad_fn=None, grad=False>"
    assert str(c._gradient) == "<Scalar(1.0), grad_fn=None, grad=False>"
    return