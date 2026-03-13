import numpy as np

class Layer():
    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    