import numpy as np

class Loss:
    def forward(y_hat, y):
        raise NotImplementedError
    
    def backward(y_hat : np.ndarray, y: np.ndarray):
        raise NotImplementedError
    
# def MSE(y_hat, y):
#     loss = np.mean(np.square(y_hat - y))
#     return loss

# def BCE(y_hat, y):
#     loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
#     return loss 