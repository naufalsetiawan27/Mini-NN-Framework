import numpy as np
from .loss import *

class MSE(Loss):
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
       L = np.mean(np.square(y_hat - y))
       return L
    
    def backward(self, y_hat : np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = y.shape[0]

        grad = 2*(y_hat - y) / batch_size
        return grad