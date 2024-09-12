from random import random
from typing import Tuple

from terox.tensor import Tensor
from terox.module import Module, Parameter

class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.input_size)] for _ in range(self.hidden_size)]))
        self.W_hh = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]))
        if self.bias:
            self.b_ih = Parameter(Tensor([2 * (random() - 0.5) for _ in range(self.hidden_size)]))
            self.b_hh = Parameter(Tensor([2 * (random() - 0.5) for _ in range(self.hidden_size)]))
        return
        
    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        h_t = input @ self.W_ih.value() + hidden @ self.W_hh.value()
        if self.bias:
            h_t = h_t + self.b_ih.value() + self.b_hh.value()
        h_t = Tensor.tanh(h_t)
        return h_t
    
class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.input_size)] for _ in range(4 * self.hidden_size)]))
        self.W_hh = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.hidden_size)] for _ in range(4 * self.hidden_size)]))
        if self.bias:
            self.b_ih = Parameter(Tensor([2 * (random() - 0.5) for _ in range(4 * self.hidden_size)]))
            self.b_hh = Parameter(Tensor([2 * (random() - 0.5) for _ in range(4 * self.hidden_size)]))
        return
        
    def forward(self, input: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h_t, c_t = hidden
        gates = input @ self.W_ih.value() + h_t @ self.W_hh.value()
        if self.bias:
            gates += self.b_ih.value() + self.b_hh.value()
        gate_size = self.hidden_size
        i_t = Tensor.sigmoid(gates[:, :gate_size])
        f_t = Tensor.sigmoid(gates[:, gate_size:2*gate_size])
        g_t = Tensor.tanh(gates[:, 2*gate_size:3*gate_size])
        o_t = Tensor.sigmoid(gates[:, 3*gate_size:])
        c_next = f_t * c_t + i_t * g_t
        h_next = o_t * Tensor.tanh(c_next)
        return h_next, c_next

class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.input_size)] for _ in range(3 * self.hidden_size)]))
        self.W_hh = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.hidden_size)] for _ in range(3 * self.hidden_size)]))
        if self.bias:
            self.b_ih = Parameter(Tensor([2 * (random() - 0.5) for _ in range(3 * self.hidden_size)]))
            self.b_hh = Parameter(Tensor([2 * (random() - 0.5) for _ in range(3 * self.hidden_size)]))
        return
        
    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        gates = input @ self.W_ih.value() + hidden @ self.W_hh.value()
        if self.bias:
            gates += self.b_ih.value() + self.b_hh.value()
        gate_size = self.hidden_size
        z_t = Tensor.sigmoid(gates[:, :gate_size])
        r_t = Tensor.sigmoid(gates[:, gate_size:2*gate_size])
        h_tilda_t = Tensor.tanh(r_t * (hidden @ self.W_hh.value()[:, 2*gate_size:]) + input @ self.W_ih.value()[:, 2*gate_size:])
        h_next = (1 - z_t) * hidden + z_t * h_tilda_t
        return h_next
