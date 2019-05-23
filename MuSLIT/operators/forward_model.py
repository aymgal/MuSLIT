__author__ = 'aymgal'


import numpy as np



class ForwardModel(object):
    """Class that describes a the forward operator HA·F in the model Y = H * A * X * F + N
    and its transpose operator (for gradient calculations).
    It also allows to compute its Lipschitz constant.
    """

    def __init__(self, blurring_matrix, mixing_matrix, lensing_martrix):

        self.F = 


    def _apply_HAF(self, X):
        return self.H(self.A(self(F(component_matrix))))

    def operator(self, component_matrix):
        return self._apply_HAF(component_matrix)

    def transpose(self):


    def lipschitz_constant(self):
        Lip = self._power_method()
        return Lip

    def _power_method(self):
        X_l = LightComponent(self.num_pix, random_init=True)
        X_s = LightComponent(self.num_pix_src, random_init=True)
        X = ComponentMatrix([X_l.data, X_s.data])
        for _ in range(n_iter):
           x = x / np.linalg.norm(x, 2)
           x = A_T(A(x))
        return np.linalg.norm(x, 2)
