import numpy as np
from NeuralNetwork import MLP
from Layer import FC
from Activation import Sigmoid, ReLU
from Loss import *


def main():
    # rng = np.random.default_rng()

    # data
    X = np.array([[1.0, 0.5, 0.2, 0.8], 
                  [0.9, 0.6, 0.3, 0.7]])
    n_features = X.shape[1]

    Y = np.array([[1.0],
                 [0.9]])

    # model
    model = MLP([FC(n_features,1),
                 ReLU()])
    logits = model.forward_pass(X)
    loss = MSE(logits, Y)

    return logits, loss

output, loss = main()
print(f"logits: {output}")
print(f"loss: {loss}")
