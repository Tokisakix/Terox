import numpy as np
from numpy.typing import NDArray

from typing import Iterable, Tuple

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

def conv1d_forward(a: np.ndarray, w: np.ndarray, stride: int, padding: int) -> np.ndarray:
    batch_size, in_channels, input_length = a.shape
    out_channels, in_channels, kernel_size = w.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    output = np.zeros((batch_size, out_channels, output_length))
    for i in range(output_length):
        input_slice = a_padded[:, :, i * stride : i * stride + kernel_size]
        output[:, :, i] = np.tensordot(input_slice, w, axes=((1, 2), (1, 2)))
    return output

def conv1d_backward(a: np.ndarray, w: np.ndarray, stride: int, padding: int, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, in_channels, input_length = a.shape
    out_channels, in_channels, kernel_size = w.shape
    a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    grad_a = np.zeros_like(a_padded)
    grad_w = np.zeros_like(w)
    output_length = grad.shape[-1]
    for i in range(output_length):
        input_slice = a_padded[:, :, i * stride : i * stride + kernel_size]
        grad_w += np.tensordot(grad[:, :, i], input_slice, axes=(0, 0))
        grad_a[:, :, i * stride : i * stride + kernel_size] += np.tensordot(grad[:, :, i], w, axes=(1, 0))
    grad_a = grad_a[:, :, padding:input_length + padding]
    return grad_a, grad_w

def conv2d_forward(a: np.ndarray, w: np.ndarray, stride: int, padding: int) -> np.ndarray:
    batch_size, in_channels, input_height, input_width = a.shape
    out_channels, in_channels, kernel_height, kernel_width = w.shape
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    output = np.zeros((batch_size, out_channels, output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            input_slice = a_padded[:, :, i * stride : i * stride + kernel_height, j * stride : j * stride + kernel_width]
            output[:, :, i, j] = np.tensordot(input_slice, w, axes=((1, 2, 3), (1, 2, 3)))
    return output

def conv2d_backward(a: np.ndarray, w: np.ndarray, stride: int, padding: int, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, in_channels, input_height, input_width = a.shape
    out_channels, in_channels, kernel_height, kernel_width = w.shape
    a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    grad_a = np.zeros_like(a_padded)
    grad_w = np.zeros_like(w)
    output_height = grad.shape[-2]
    output_width = grad.shape[-1]
    for i in range(output_height):
        for j in range(output_width):
            input_slice = a_padded[:, :, i * stride : i * stride + kernel_height, j * stride : j * stride + kernel_width]
            grad_w += np.tensordot(grad[:, :, i, j], input_slice, axes=(0, 0))
            grad_a[:, :, i * stride : i * stride + kernel_height, j * stride : j * stride + kernel_width] += np.tensordot(grad[:, :, i, j], w, axes=(1, 0))
    grad_a = grad_a[:, :, padding:input_height + padding, padding:input_width + padding]
    return grad_a, grad_w

def conv3d_forward(a: np.ndarray, w: np.ndarray, stride: int, padding: int) -> np.ndarray:
    batch_size, in_channels, input_depth, input_height, input_width = a.shape
    out_channels, in_channels, kernel_depth, kernel_height, kernel_width = w.shape
    output_depth = (input_depth + 2 * padding - kernel_depth) // stride + 1
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding), (padding, padding)), mode='constant')
    output = np.zeros((batch_size, out_channels, output_depth, output_height, output_width))
    for d in range(output_depth):
        for i in range(output_height):
            for j in range(output_width):
                input_slice = a_padded[:, :, d * stride : d * stride + kernel_depth, i * stride : i * stride + kernel_height, j * stride : j * stride + kernel_width]
                output[:, :, d, i, j] = np.tensordot(input_slice, w, axes=((1, 2, 3, 4), (1, 2, 3, 4)))
    return output

def conv3d_backward(a: np.ndarray, w: np.ndarray, stride: int, padding: int, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, in_channels, input_depth, input_height, input_width = a.shape
    out_channels, in_channels, kernel_depth, kernel_height, kernel_width = w.shape
    a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding), (padding, padding)), mode='constant')
    grad_a = np.zeros_like(a_padded)
    grad_w = np.zeros_like(w)
    output_depth = grad.shape[-3]
    output_height = grad.shape[-2]
    output_width = grad.shape[-1]
    for d in range(output_depth):
        for i in range(output_height):
            for j in range(output_width):
                input_slice = a_padded[:, :, d * stride : d * stride + kernel_depth, i * stride : i * stride + kernel_height, j * stride : j * stride + kernel_width]
                grad_w += np.tensordot(grad[:, :, d, i, j], input_slice, axes=(0, 0))
                grad_a[:, :, d * stride : d * stride + kernel_depth, i * stride : i * stride + kernel_height, j * stride : j * stride + kernel_width] += np.tensordot(grad[:, :, d, i, j], w, axes=(1, 0))
    grad_a = grad_a[:, :, padding:input_depth + padding, padding:input_height + padding, padding:input_width + padding]
    return grad_a, grad_w