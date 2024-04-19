import pytest
import numpy as np
from numpy.typing import NDArray

from terox.tensor.tensor import Tensor
from terox.tensor.tensor_opts import Mean, Permute, Reshape, Sum, TensorOptsBackend
from terox.tensor.tensor_opts import Add, Sub, Mul, Div, Neg, Eq, Lt, Gt, Abs, Exp, Log, Relu, Sigmoid, Tanh

Tensor_opts_backend = TensorOptsBackend()

@pytest.mark.test_tensor_opts
def test_add(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Add(A, B)
    assert (A.item() + B.item()).all() == C.item().all()
    assert C._history._func.__class__ == Add
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Add(A, B)
    assert (A.item() + B.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_sub(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Sub(A, B)
    assert (A.item() - B.item()).all() == C.item().all()
    assert C._history._func.__class__ == Sub
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Sub(A, B)
    assert (A.item() - B.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_mul(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Mul(A, B)
    assert (A.item() * B.item()).all() == C.item().all()
    assert C._history._func.__class__ == Mul
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Mul(A, B)
    assert (A.item() * B.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_div(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Div(A, B)
    assert (A.item() / B.item()).all() == C.item().all()
    assert C._history._func.__class__ == Div
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Div(A, B)
    assert (A.item() / B.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_neg(a:NDArray=[[1.0, 2.0], [3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Neg(A)
    assert (-A.item()).all() == C.item().all()
    assert C._history._func.__class__ == Neg
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Neg(A)
    assert (-A.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_eq(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Eq(A, B)
    res = np.zeros_like(a)
    res[np.array(a) == np.array(b)] = 1.0
    assert res.all() == C.item().all()
    assert C._history._func.__class__ == Eq
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Eq(A, B)
    assert res.all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_lt(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Lt(A, B)
    res = np.zeros_like(a)
    res[np.array(a) < np.array(b)] = 1.0
    assert res.all() == C.item().all()
    assert C._history._func.__class__ == Lt
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Lt(A, B)
    assert res.all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_gt(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = Tensor_opts_backend.Gt(A, B)
    res = np.zeros_like(a)
    res[np.array(a) > np.array(b)] = 1.0
    assert res.all() == C.item().all()
    assert C._history._func.__class__ == Gt
    assert C._history._args == (A, B)
    A = Tensor(a, _require_grad=False)
    B = Tensor(b, _require_grad=False)
    C = Tensor_opts_backend.Gt(A, B)
    assert res.all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_abs(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Abs(A)
    assert abs(A.item()).all() == C.item().all()
    assert C._history._func.__class__ == Abs
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Abs(A)
    assert abs(A.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_exp(a:NDArray=[[1.0, 2.0], [3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Exp(A)
    assert np.exp(A.item()).all() == C.item().all()
    assert C._history._func.__class__ == Exp
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Exp(A)
    assert np.exp(A.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_log(a:NDArray=[[1.0, 2.0], [3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Log(A)
    assert np.log(A.item()).all() == C.item().all()
    assert C._history._func.__class__ == Log
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Log(A)
    assert np.log(A.item()).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_relu(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Relu(A)
    res = np.array(a)
    res[res <= 0.0] = 0.0
    assert res.all() == C.item().all()
    assert C._history._func.__class__ == Relu
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Relu(A)
    assert res.all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_sigmoid(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Sigmoid(A)
    assert (1.0 / (1.0 + np.exp(-A.item()))).all() == C.item().all()
    assert C._history._func.__class__ == Sigmoid
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Sigmoid(A)
    assert (1.0 / (1.0 + np.exp(-A.item()))).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_tanh(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Tanh(A)
    assert ((np.exp(A.item()) - np.exp(-A.item())) / (np.exp(A.item()) + np.exp(-A.item()))).all() == C.item().all()
    assert C._history._func.__class__ == Tanh
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Tanh(A)
    assert ((np.exp(A.item()) - np.exp(-A.item())) / (np.exp(A.item()) + np.exp(-A.item()))).all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_reshape(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Reshape(A, (4, 1))
    assert Tensor([[-1.0], [-2.0], [-3.0], [-4.0]]).item().all() == C.item().all()
    assert C._history._func.__class__ == Reshape
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Reshape(A, (4, 1))
    assert Tensor([[-1.0], [-2.0], [-3.0], [-4.0]]).item().all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_permute(a:NDArray=[[-1.0, -2.0, -3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Permute(A, (0, 1))
    assert Tensor([[-1.0], [-2.0], [-3.0], [-4.0]]).item().all() == C.item().all()
    assert C._history._func.__class__ == Permute
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Permute(A, (0, 1))
    assert Tensor([[-1.0], [-2.0], [-3.0], [-4.0]]).item().all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_sum(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Sum(A, 1)
    assert Tensor([[-4.0], [-6.0]]).item().all() == C.item().all()
    assert C._history._func.__class__ == Sum
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Sum(A, 1)
    assert Tensor([[-4.0], [-6.0]]).item().all() == C.item().all()
    assert C._history == None
    return

@pytest.mark.test_tensor_opts
def test_mean(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = Tensor_opts_backend.Mean(A, 1)
    assert Tensor([[-2.0], [-3.0]]).item().all() == C.item().all()
    assert C._history._func.__class__ == Mean
    assert C._history._args == (A,)
    A = Tensor(a, _require_grad=False)
    C = Tensor_opts_backend.Mean(A, 1)
    assert Tensor([[-2.0], [-3.0]]).item().all() == C.item().all()
    assert C._history == None
    return