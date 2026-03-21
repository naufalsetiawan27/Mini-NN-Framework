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
    
    def train(self, x: np.ndarray, 
                    y: np.ndarray, 
                    loss_function: Loss, 
                    optimizer: Optimizer,
                    epochs: int,
                    lr : float
                    # early_stopping = False,
                    # patience = 0
                    ):
        
        # if not early_stopping:
        #     patience = 0

        loss_function = self.registry_loss[loss_function]()
        self.loss_history = []

        optimizer = self.registry_opt[optimizer](lr)
        # termination conditions
        epochs_remaining = epochs
        # early_stopping = early_stopping

        while epochs_remaining >= 1 :
        # and not early_stopping:

            print(f"EPOCH:{epochs - epochs_remaining + 1}/{epochs}")
            # forward pass
            y_hat = self.forward_pass(x)

            # calculate loss
            loss = loss_function.forward(y_hat, y)
            print(loss)
            self.loss_history.append(loss)

            # backpropagation
            grad = loss_function.backward(y_hat, y)
            self.backward_pass(grad)

             # update parameters
            for layer in self.objects:
                if isinstance(layer, Layer):
                    optimizer.update_params(layer.weights, layer.dzdw)
                    optimizer.update_params(layer.bias, layer.dzdb)

            epochs_remaining -= 1

        return self