import numpy as np
from .optimizer import *

class SGD(Optimizer):
    def __init__(self, lr : float):
        self.lr = lr

    def update_params(self, params: np.ndarray, grad: np.ndarray):
        params -= self.lr * grad
        
