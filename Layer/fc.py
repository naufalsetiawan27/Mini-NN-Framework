import numpy as np

class FC():
     def __init__(self, n_input: int, n_output: int, rng = None):
         
         self.n_input = n_input
         self.n_output = n_output
         if rng == None:
              rng = np.random.default_rng()
         self.weights = rng.uniform(-5 , 5, (n_output, n_input))
         self.bias = rng.uniform(-5, 5, (n_output,))

     def forward(self, a_prev:np.ndarray) -> np.ndarray:
          # a_prev  -> shape: (batch, input_feat)
          # weight  -> shape: (n_output, n_input)
          # bias    -> shape: (n_output, 1)
          # z       -> shape : (batch, n_output)

          self.a_prev = a_prev

          # batch forward
          self.z = a_prev @ self.weights.T + self.bias.T

          return self.z 
     
     def backward(self, grad:np.ndarray) -> np.ndarray:
          # grad    -> shape:(batch, n_output)
          # weights  -> shape:(n_output, n_input)
          # dzda    -> shape:(batch, n_input)
          # a_prev  -> shape: (batch, input_feat)

          self.dzda =  grad @ self.weights

          batch_size = self.a_prev.shape[0]
          self.dzdw = grad.T @ self.a_prev/batch_size

          dzdb = grad.T
          self.dzdb = np.mean(dzdb ,axis = 1, keepdims=True)

          return self.dzda
     
     def get_params(self):
          return[]