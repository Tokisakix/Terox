import pytest
import numpy as np
from numpy.typing import NDArray

from terox.tensor.tensor import Tensor

def TensorEq(a:Tensor, b:Tensor) -> bool:
    return a.item().all() == b.item().all()

@pytest.mark.test_tensor_backward
def test_add_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A + B
    C.backward()
    assert TensorEq(A._gradient, C.one(A.shape()))
    assert TensorEq(B._gradient, C.one(B.shape()))
    return

@pytest.mark.test_tensor_backward
def test_sub_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A - B
    C.backward()
    assert TensorEq(A._gradient, C.one(A.shape()))
    assert TensorEq(B._gradient, -C.one(B.shape()))
    return

@pytest.mark.test_tensor_backward
def test_mul_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A * B
    C.backward()
    assert TensorEq(A._gradient, B)
    assert TensorEq(B._gradient, A)
    return

@pytest.mark.test_tensor_backward
def test_matmul_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A @ B
    C.backward()
    assert TensorEq(A._gradient, B.tranpose())
    assert TensorEq(B._gradient, A.tranpose())
    return

@pytest.mark.test_tensor_backward
def test_div_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A / B
    C.backward()
    assert TensorEq(A._gradient, C.one(B.shape()) / B)
    assert TensorEq(B._gradient, -A / (B * B))
    return

@pytest.mark.test_tensor_backward
def test_neg_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = -A
    C.backward()
    assert TensorEq(A._gradient, -C.one(A.shape()))
    return

@pytest.mark.test_tensor_backward
def test_eq_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A == B
    C.backward()
    assert TensorEq(A._gradient, C.zero(A.shape()))
    assert TensorEq(B._gradient, C.zero(B.shape()))
    return

@pytest.mark.test_tensor_backward
def test_lt_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A < B
    C.backward()
    assert TensorEq(A._gradient, C.zero(A.shape()))
    assert TensorEq(B._gradient, C.zero(B.shape()))
    return

@pytest.mark.test_tensor_backward
def test_gt_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]], b:NDArray=[[2.0, 3.0], [4.0, 5.0]]) -> None:
    A = Tensor(a)
    B = Tensor(b)
    C = A > B
    C.backward()
    assert TensorEq(A._gradient, C.zero(A.shape()))
    assert TensorEq(B._gradient, C.zero(B.shape()))
    return

@pytest.mark.test_tensor_backward
def test_abs_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.abs(A)
    C.backward()
    assert TensorEq(A._gradient, -C.one(A.shape()))
    return

@pytest.mark.test_tensor_backward
def test_exp_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = A.exp()
    C.backward()
    assert TensorEq(A._gradient, A.exp())
    return

@pytest.mark.test_tensor_backward
def test_log_backward(a:NDArray=[[1.0, 2.0], [3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = A.log()
    C.backward()
    assert TensorEq(A._gradient, C.one(A.shape()) / A)
    return

@pytest.mark.test_tensor_backward
def test_relu_backward(a:NDArray=[[-1.0, 2.0], [-3.0, 4.0]]) -> None:
    A = Tensor(a)
    C = A.relu()
    C.backward()
    assert TensorEq(A._gradient, Tensor([[0.0, 1.0], [0.0, 1.0]]))
    return

@pytest.mark.test_tensor_backward
def test_sigmoid_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.sigmoid()
    C.backward()
    sigmoid = 1.0 / (1.0 + np.exp(A.item()))
    assert TensorEq(A._gradient, C.new(sigmoid * (1.0 - sigmoid)))
    return

@pytest.mark.test_tensor_backward
def test_tanh_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.tanh()
    C.backward()
    tanh = (np.exp(A.item()) - np.exp(-A.item())) / (np.exp(A.item()) + np.exp(-A.item()))
    assert TensorEq(A._gradient, C.new(1.0 - tanh * tanh))
    return

@pytest.mark.test_tensor_backward
def test_reshape_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.reshape((4, 1))
    C.backward()
    assert TensorEq(A._gradient, C.new([[1.0], [1.0], [1.0], [1.0]]))
    return

@pytest.mark.test_tensor_backward
def test_permute_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.permute((0, 1))
    C.backward()
    assert TensorEq(A._gradient, C.new([[1.0], [1.0], [1.0], [1.0]]))
    return

@pytest.mark.test_tensor_backward
def test_sum_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.sum(1)
    C.backward()
    assert TensorEq(A._gradient, C.new([[1.0, 1.0], [1.0, 1.0]]))
    return

@pytest.mark.test_tensor_backward
def test_mean_backward(a:NDArray=[[-1.0, -2.0], [-3.0, -4.0]]) -> None:
    A = Tensor(a)
    C = A.mean(1)
    C.backward()
    assert TensorEq(A._gradient, C.new([[1.0, 1.0], [1.0, 1.0]]))
    return

@pytest.mark.test_tensor_backward
def test_complex_backward() -> None:
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[2.0, 3.0], [4.0, 5.0]])
    C = A + B
    D = A + C
    E = (D - C) * (D - C)
    E.backward()
    assert TensorEq(A._gradient, Tensor([[2.0, 4.0], [6.0, 8.0]]))
    assert TensorEq(B._gradient, Tensor([[0.0, 0.0], [0.0, 0.0]]))
    assert TensorEq(C._gradient, Tensor([[0.0, 0.0], [0.0, 0.0]]))
    assert TensorEq(D._gradient, Tensor([[2.0, 4.0], [6.0, 8.0]]))
    return

@pytest.mark.test_tensor_backward
def test_scalar_str() -> None:
    a = Tensor(1.5)
    b = Tensor(2.5)
    c = a * b
    c.backward()
    assert str(a._gradient) == "<Tensor([2.5]), grad_fn=None, grad=False>"
    assert str(c._gradient) == "<Tensor([1.]), grad_fn=None, grad=False>"
    return