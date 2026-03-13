import numpy as np
from .activation import *

class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.a =  np.maximum(0, z)
        return self.a
     
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(self.a > 0, 1, 0) * grad