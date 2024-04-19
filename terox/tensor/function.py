import numpy as np
from numpy.typing import NDArray

from typing import Iterable

def add(a:NDArray, b:NDArray) -> NDArray:
    res = a + b
    return res

def sub(a:NDArray, b:NDArray) -> NDArray:
    res = a - b
    return res

def mul(a:NDArray, b:NDArray) -> NDArray:
    res = a * b
    return res

def tranpose(a:NDArray) -> NDArray:
    res = a.transpose()
    return res

def matmul(a:NDArray, b:NDArray) -> NDArray:
    res = a @ b
    return res

def div(a:NDArray, b:NDArray) -> NDArray:
    res = a / b
    return res

def neg(a:NDArray) -> NDArray:
    res = -a
    return res

def eq(a:NDArray, b:NDArray) -> NDArray:
    res = np.zeros_like(a)
    res[a == b] = 1.0
    return res

def lt(a:NDArray, b:NDArray) -> NDArray:
    res = np.zeros_like(a)
    res[a < b] = 1.0
    return res

def gt(a:NDArray, b:NDArray) -> NDArray:
    res = np.zeros_like(a)
    res[a > b] = 1.0
    return res

def abs(a:NDArray) -> NDArray:
    res = np.abs(a)
    return res

def exp(a:NDArray) -> NDArray:
    res = np.exp(a)
    return res

def log(a:NDArray) -> NDArray:
    res = np.log(a)
    return res

def relu(a:NDArray) -> NDArray:
    res = np.copy(a)
    res[a <= 0.0] = 0.0
    return res

def reshape(a:NDArray, _shape:Iterable) -> NDArray:
    res = np.reshape(a, _shape)
    return res

def permute(a:NDArray, order:Iterable) -> NDArray:
    res = np.transpose(a, order)
    return res