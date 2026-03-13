import numpy as np

class Activation():
    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    


