from random import random

from terox.tensor import Tensor
from terox.module import Module, Parameter

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor([[2 * (random() - 0.5) for _ in range(self.num_embeddings)] for _ in range(self.embedding_dim)]))
        return
    
    def forward(self, indices: Tensor) -> Tensor:
        return self.weight.value()[indices._item]