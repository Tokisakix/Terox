import pytest
import math

from terox.autodiff.scalar import Scalar
from terox.autodiff.scalar_opts import ScalarOptsBackend
from terox.autodiff.scalar_opts import Add, Sub, Mul, Div, Neg, Max, Min, Eq, Lt, Gt, Abs, Exp, Log, Relu, Sigmoid, Tanh

scalar_opts_backend = ScalarOptsBackend()

@pytest.mark.test_scalar_opts
def test_add(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Add(A, B)
    assert A.item() + B.item() == C.item()
    assert C._history._func.__class__ == Add
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Add(A, B)
    assert A.item() + B.item() == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_sub(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Sub(A, B)
    assert A.item() - B.item() == C.item()
    assert C._history._func.__class__ == Sub
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Sub(A, B)
    assert A.item() - B.item() == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_mul(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Mul(A, B)
    assert A.item() * B.item() == C.item()
    assert C._history._func.__class__ == Mul
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Mul(A, B)
    assert A.item() * B.item() == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_div(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Div(A, B)
    assert A.item() / B.item() == C.item()
    assert C._history._func.__class__ == Div
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Div(A, B)
    assert A.item() / B.item() == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_neg(a:float=2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Neg(A)
    assert -A.item() == C.item()
    assert C._history._func.__class__ == Neg
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Neg(A)
    assert -A.item() == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_max(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Max(A, B)
    assert max(A.item(), B.item()) == C.item()
    assert C._history._func.__class__ == Max
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Max(A, B)
    assert max(A.item(), B.item()) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_min(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Min(A, B)
    assert min(A.item(), B.item()) == C.item()
    assert C._history._func.__class__ == Min
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Min(A, B)
    assert min(A.item(), B.item()) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_eq(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Eq(A, B)
    assert (1.0 if A.item() == B.item() else 0.0) == C.item()
    assert C._history._func.__class__ == Eq
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Eq(A, B)
    assert (1.0 if A.item() == B.item() else 0.0) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_lt(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Lt(A, B)
    assert (1.0 if A.item() < B.item() else 0.0) == C.item()
    assert C._history._func.__class__ == Lt
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Lt(A, B)
    assert (1.0 if A.item() < B.item() else 0.0) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_gt(a:float=2.0, b:float=3.0) -> None:
    A = Scalar(a)
    B = Scalar(b)
    C = scalar_opts_backend.Gt(A, B)
    assert (1.0 if A.item() > B.item() else 0.0) == C.item()
    assert C._history._func.__class__ == Gt
    assert C._history._args == (A, B)
    A = Scalar(a, _require_grad=False)
    B = Scalar(b, _require_grad=False)
    C = scalar_opts_backend.Gt(A, B)
    assert (1.0 if A.item() > B.item() else 0.0) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_abs(a:float=-2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Abs(A)
    assert abs(A.item()) == C.item()
    assert C._history._func.__class__ == Abs
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Abs(A)
    assert abs(A.item()) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_exp(a:float=2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Exp(A)
    assert math.exp(A.item()) == C.item()
    assert C._history._func.__class__ == Exp
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Exp(A)
    assert math.exp(A.item()) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_log(a:float=2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Log(A)
    assert math.log(A.item()) == C.item()
    assert C._history._func.__class__ == Log
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Log(A)
    assert math.log(A.item()) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_relu(a:float=-2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Relu(A)
    assert (A.item() if A.item() >= 0.0 else 0.0) == C.item()
    assert C._history._func.__class__ == Relu
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Relu(A)
    assert (A.item() if A.item() >= 0.0 else 0.0) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_sigmoid(a:float=-2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Sigmoid(A)
    assert 1 / (1.0 + math.exp(-A.item())) == C.item()
    assert C._history._func.__class__ == Sigmoid
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Sigmoid(A)
    assert 1 / (1.0 + math.exp(-A.item())) == C.item()
    assert C._history == None
    return

@pytest.mark.test_scalar_opts
def test_tanh(a:float=-2.0) -> None:
    A = Scalar(a)
    C = scalar_opts_backend.Tanh(A)
    assert (math.exp(A.item()) - math.exp(-A.item())) / (math.exp(A.item()) + math.exp(-A.item())) == C.item()
    assert C._history._func.__class__ == Tanh
    assert C._history._args == (A,)
    A = Scalar(a, _require_grad=False)
    C = scalar_opts_backend.Tanh(A)
    assert (math.exp(A.item()) - math.exp(-A.item())) / (math.exp(A.item()) + math.exp(-A.item())) == C.item()
    assert C._history == None
    return