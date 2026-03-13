import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import MLP
from Layer import *
from Activation import Sigmoid, ReLU
from Loss import MSE
from Optimizer import SGD

def main():
    # data
    np.random.seed(42)

    X = np.random.rand(100, 4)

    Y = np.random.rand(100, 1) 

    n_features = X.shape[1]

    # model
    model = MLP([FullyConnected(n_features,n_features),
                 ReLU(),
                 FullyConnected(n_features,1)])
    
    print(f"w_0 = {model.objects[0].weights}")

    loss_func = MSE()
    opt = SGD(lr = 0.1)
    epochs = 100
    losses = []
    for _ in range(epochs):
        print(f"EPOCH: {_+1}/{epochs}")

        # forward pass
        logits = model.forward_pass(X)

        # calculate loss
        loss = loss_func.forward(logits, Y)
        print(f"loss= {loss}")
        losses.append(loss)

        # backpropagation
        grad = loss_func.backward(logits, Y)
        model.backward_pass(grad)

        # update parameters
        for layer in model.objects:
            if isinstance(layer, Layer):
                opt.update_params(layer.weights, layer.dzdw)
                opt.update_params(layer.bias, layer.dzdb)

    print(f"w_1 = {model.objects[0].weights}")

    plt.plot(losses)
    plt.show()
    return logits

main()
