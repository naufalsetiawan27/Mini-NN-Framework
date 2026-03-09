import numpy as np
from .activation import *

class ReLU(Activation):
    def forward(self, z) -> np.ndarray:
        a =  np.maximum(0, z)
        return a 