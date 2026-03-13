import numpy as np

class Activation():
    def forward(self, z:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    


