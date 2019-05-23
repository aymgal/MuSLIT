__author__ = 'aymgal'


import numpy as np

from MuSLIT.transforms.data_structures import ComponentMatrix


def power_method_op(A, A_T, N, n_iter=25):
    x = np.random.randn(N, 1)
    for _ in range(n_iter):
       x = x / np.linalg.norm(x, 2)
       x = A_T(A(x))
    return np.linalg.norm(x, 2)

def power_method_op_compmatrix(A, A_T, N, n_iter=25):
    U0_l = LightComponent(self.num_pix, random_init=True)
    U0_s = LightComponent(self.num_pix_src, random_init=True)
    self.U0 = ComponentMatrix([U0_l.data, U0_s.data])
    
    for _ in range(n_iter):
       x = x / np.linalg.norm(x, 2)
       x = A_T(A(x))
    return np.linalg.norm(x, 2)

def conjugate(X):
    return np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(X))))

def convolve(X, H):
    return scp.fftconvolve(X, H, mode='same')