import numpy as np

import src.alc as alc
from src.headLayer import fully_connected_linear_qr

LATENT_FACTORS_SIZE = 50
NUM_CASES = 200

class TestQRMethod:

    def load_data(self):
        X = np.random.rand(LATENT_FACTORS_SIZE, NUM_CASES)
        Y = np.random.rand(2, NUM_CASES)
        return X, Y

    def test_modulo_qr_decomposition(self):
        X, _ = self.load_data()
        Q, R, _ = alc.QR_con_GS_reducido(X.T)
        assert Q.shape == (NUM_CASES, LATENT_FACTORS_SIZE), "Q matrix has incorrect shape"
        assert R.shape == (LATENT_FACTORS_SIZE, LATENT_FACTORS_SIZE), "R matrix has incorrect shape"
    
    def test_modulo_qr_decomposition_householder(self):
        X, _ = self.load_data()
        Q, R = alc.QR_con_HH(X.T)
        assert Q.shape == (NUM_CASES, LATENT_FACTORS_SIZE), "Q matrix has incorrect shape"
        assert R.shape == (LATENT_FACTORS_SIZE, LATENT_FACTORS_SIZE), "R matrix has incorrect shape"

    def test_fully_connected_linear_qr_schmidt(self):
        X, Y = self.load_data()
        W_gs = fully_connected_linear_qr(X, Y, algorithm="gs")
        assert W_gs.shape == (2, LATENT_FACTORS_SIZE), "W matrix from GS has incorrect shape"

    def test_fully_connected_linear_qr_householder(self):
        X, Y = self.load_data()
        W_hh = fully_connected_linear_qr(X, Y, algorithm="hh")
        assert W_hh.shape == (2, LATENT_FACTORS_SIZE), "W matrix from HH has incorrect shape"
