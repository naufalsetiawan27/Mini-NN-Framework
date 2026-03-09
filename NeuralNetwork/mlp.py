import numpy as np
from .neuralNetwork import *

class MLP(NeuralNetwork):
    def __init__(self, objects: list[object]):
        self.objects = objects
    
    def forward_pass(self, input: np.ndarray) -> np.ndarray: 

        output = input

        for object in self.objects:
            output = object.forward(output)

        return output
