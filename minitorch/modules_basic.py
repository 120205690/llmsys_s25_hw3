"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np
import math
from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN YOUR SOLUTION
        weight_npy = np.random.uniform(-1, 1, (num_embeddings, embedding_dim))
        self.weights = Parameter(tensor_from_numpy(weight_npy, backend = backend, requires_grad = True), name = 'embedding_weight')

        ### END YOUR SOLUTION
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        one_hot_x = one_hot(input = x, num_classes = self.num_embeddings)
        one_hot_x = one_hot_x.view(bs * seq_len, self.num_embeddings)

        output = one_hot_x @ self.weights.value
        output = output.view(bs, seq_len, self.embedding_dim)

        return output
        ### END YOUR SOLUTION

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        mask = rand(x.shape, backend = x.backend)
        if self.training == False or self.p_dropout == 0:
            return x
        mask = self.p_dropout < mask
        #adding 0 gets rid of the discrepancy between -0 and +0 
        out = mask * x * (1-self.p_dropout) + 0
        return out
        ### END YOUR SOLUTION


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN YOUR SOLUTION
        weight_npy = np.random.uniform(-1/(in_size**0.5), 1/(in_size**0.5), (in_size, out_size))
        bias_npy = np.random.uniform(-1/(in_size**0.5), 1/(in_size**0.5), (out_size,))
        
        self.weights = Parameter(tensor_from_numpy(weight_npy, backend = backend, requires_grad = True), name = 'linear_weight')
        if bias:
            self.bias = Parameter(tensor_from_numpy(bias_npy, backend = backend, requires_grad = True), name = 'linear_bias')
        else:
            bias = None
        # raise NotImplementedError
        # Uniform(-sqrt(1/in_features), sqrt(1/in_features)) 
        ### END YOUR SOLUTION

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        # batch, in_size = x.shape
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError

        output =  x @ self.weights.value
        if self.bias:
            output += self.bias.value
        return output
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        self.weights = Parameter(tensor(([1]*self.dim), backend = backend, requires_grad = True), name = 'layerNorm_weight')
        self.bias = Parameter(tensor(([0]*self.dim), backend = backend, requires_grad = True), name = 'layerNorm_bias')
        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        mean = x.mean(dim = 1)
        var = x.var(dim = 1)
        x_norm = (x-mean)/((var + self.eps).__pow__(0.5))
        out = x_norm * self.weights.value
        out = out + self.bias.value
        return out
        ### END YOUR SOLUTION
