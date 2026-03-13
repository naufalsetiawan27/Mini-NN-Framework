import numpy as np
from .layer import *

class FullyConnected(Layer):
     def __init__(self, n_input: int, n_output: int, rng = None):
          self.n_input = n_input
          self.n_output = n_output

          # initialize parameters
          if rng == None:
               rng = np.random.default_rng()
          limit = 1 / np.sqrt(n_input)
          self.weights = rng.uniform(-limit , limit, (n_output, n_input))
          self.bias = np.zeros((n_output, 1))

     def forward(self, a_prev:np.ndarray) -> np.ndarray:
          # a_prev  -> shape: (batch, n_input)
          # weight  -> shape: (n_output, n_input)
          # bias    -> shape: (n_output, 1)
          # z       -> shape : (batch, n_output)

          self.a_prev = a_prev
          # batch forward
          self.z = a_prev @ self.weights.T + self.bias.T

          return self.z 
     
     def backward(self, grad:np.ndarray) -> np.ndarray:
          # grad    -> shape:(batch, n_output)
          # weights  -> shape:(n_output, n_input)
          # dzda    -> shape:(batch, n_input)
          # a_prev  -> shape: (batch, input_feat)

          self.dzda =  grad @ self.weights # dz/da

          batch_size = self.a_prev.shape[0]
          self.dzdw = grad.T @ self.a_prev/batch_size # dz/dw

          dzdb = grad.T
          self.dzdb = np.mean(dzdb ,axis = 1, keepdims=True) # dz/db

          return self.dzda