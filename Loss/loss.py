import numpy as np

def MSE(y_hat, y):
    loss = np.mean(np.square(y_hat - y))
    return loss

def BCE(y_hat, y):
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss 