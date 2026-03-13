import numpy as np
from NeuralNetwork import MLP
from Layer import FC
from Activation import Sigmoid, ReLU
from Loss import MSE


def main():
    # rng = np.random.default_rng()

    # data
    X = np.array([[1.0, 0.5, 0.2, 0.8], 
                  [0.9, 0.6, 0.3, 0.7]])
    n_features = X.shape[1]

    Y = np.array([[1.0],
                 [0.9]])

    # model
    model = MLP([FC(n_features,n_features),
                 ReLU(),
                 FC(n_features,1),
                 Sigmoid()])
    print(f"w0 = {model.objects[0].weights}")
    logits = model.forward_pass(X)
    print(f"logits= {logits}")
    loss_func = MSE()
    loss = loss_func.forward(logits, Y)
    print(f"loss= {loss}")
    grad = loss_func.backward(logits, Y)
    print(f"grad= {grad}")
    model.backward_pass(grad)

    return logits

main()
# output, loss = main()
# print(f"logits: {output}")
# print(f"loss: {loss}")
