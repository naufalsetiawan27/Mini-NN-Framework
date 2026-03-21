import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import MLP
from Layer import *
from Activation import Sigmoid, ReLU
from Loss import MSE
from Optimizer import SGD
from Dropout import Dropout

def main():
    # data
    np.random.seed(42)

    X = np.random.rand(100, 4)

    Y = np.random.rand(100, 1) 

    n_features = X.shape[1]

    # model
    model = MLP([FullyConnected(n_features,n_features),
                 ReLU(),
                #  Dropout(prob = 0.5),
                 FullyConnected(n_features,n_features),
                 ReLU(),
                #  Dropout(prob = 0.5),
                 FullyConnected(n_features,1)])
    
    print(f"w_0 = {model.objects[0].weights}")

    model.train(
        X, 
        Y, 
        loss_function= "mse",
        optimizer= "sgd",
        epochs=100,
        lr = 0.2
    )
    print(f"w_1 = {model.objects[0].weights}")

    plt.plot(model.loss_history)
    plt.show()

main()
