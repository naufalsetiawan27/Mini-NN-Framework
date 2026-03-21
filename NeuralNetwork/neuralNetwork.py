import numpy as np
from Loss import *
from Optimizer import *
from Layer import *

class NeuralNetwork():
    registry_loss = {
        "mse" : MSE
    }
    registry_opt = {
        "sgd" : SGD 
    }

    def __init__(self):
        raise NotImplementedError
    
    def forward_pass(self, input: np.ndarray) -> np.ndarray: 
        raise NotImplementedError
    
    def backward_pass(self, input: np.ndarray) -> np.ndarray: 
        raise NotImplementedError
    
