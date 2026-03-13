import numpy as np

class Optimizer:
    def update_params(self, params: np.ndarray, grad: np.ndarray):
        raise NotImplementedError