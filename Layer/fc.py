import numpy as np

class FC():
    def __init__(self, n_input: int, n_output: int, rng = None):
         
         self.n_input = n_input
         self.n_output = n_output
         if rng == None:
              rng = np.random.default_rng()
         self.weights = rng.uniform(-5 , 5, (n_output, n_input))
         self.bias = rng.uniform(-5, 5, (n_output, 1))

    def forward(self, input:np.ndarray) -> np.ndarray:
          # z = self.weights @ input + self.bias
          
          # batch forward
          z = input @ self.weights.T + self.bias
          return z