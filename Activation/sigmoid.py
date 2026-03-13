import numpy as np
from .activation import *

class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.a = 1 / (1 + np.exp(-z))
        return self.a
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (self.a * (1- self.a)) * grad