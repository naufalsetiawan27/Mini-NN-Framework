import numpy as np
from .neuralNetwork import *

class MLP(NeuralNetwork):
    def __init__(self, objects: list[object]):
        self.objects = objects
    
    def forward_pass(self, inp: np.ndarray) -> np.ndarray: 
        out = inp

        for object in self.objects:
            out = object.forward(out)

        return out

    def backward_pass(self, grad: np.ndarray) -> np.ndarray:

        for object in reversed(self.objects):
            grad = object.backward(grad)