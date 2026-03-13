import numpy as np

class Loss:
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError
    
    def backward(self, y_hat : np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
