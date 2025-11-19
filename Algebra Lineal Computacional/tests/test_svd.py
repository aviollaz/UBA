import numpy as np
from src.headLayer import fully_connected_lineal_svd

LATENT_FACTORS_SIZE = 100
NUM_CASES = 500

class TestSVDMethod:

    def load_data(self):
        X = np.random.rand(LATENT_FACTORS_SIZE, NUM_CASES)
        Y = np.random.rand(2, NUM_CASES)
        return X, Y

    def fully_connected_lineal_svd(self):
        X, Y = self.load_data()
        W_hh = fully_connected_lineal_svd(X, Y, algorithm="hh")
        assert W_hh.shape == (2, LATENT_FACTORS_SIZE), "W matrix from HH has incorrect shape"
