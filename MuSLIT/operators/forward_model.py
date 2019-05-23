__author__ = 'aymgal'


import numpy as np



class ForwardModel(object):
    """Class that describes a the forward operator HA·F in the model Y = H * A * X * F + N
    and its transpose operator (for gradient calculations).
    It also allows to compute its Lipschitz constant.
    """

    def __init__(self, blurring_matrix, mixing_matrix, lensing_martrix):
        self.H   = blurring_matrix.H
        self.H_t = blurring_matrix.H_t
        self.A   = mixing_matrix.A
        self.A_t = mixing_matrix.A_t
        self.F   = lensing_martrix.F
        self.F_t = lensing_martrix.F_t

    def _apply_HAF(self, X):
        """here X must be a ComponentMatrix object"""
        return self.H(self.A(self(F(X))))

    def _apply_FAH(self, X):
        """here X must be a numpy array of shape (num_bands, num_pix**2)"""
        return self.F_t(self.A_t(self(H_t(X))))

    def operator(self, component_matrix):
        return self._apply_HAF(component_matrix)

    def transpose(self, multiband_image):
        return self._apply_FAH(multiband_image)

    def lipschitz(self):
        if not hasattr(self, '_Lip')
            self._Lip = self._power_method()
        return self._Lip

    def _power_method(self):
        X_l = LightComponent(self.num_pix, random_init=True)
        X_s = LightComponent(self.num_pix_src, random_init=True)
        X = ComponentMatrix([X_l.data, X_s.data])
        for _ in range(n_iter):
           X = 1. / X.norm(p=2) * X
           X = self.transpose(self.operator(X))
        return X.norm(p=2)
