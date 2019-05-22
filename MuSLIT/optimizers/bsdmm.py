__author__ = 'aymgal'

"""Implementation of the Block-SDMM algorithm for NMF (Moolekamp & Melchior 2017)

FOR NOW it is just a wrapper around 'proxmin' by the authors above,
forked here :https://github.com/aymgal/proxmin
"""

from proxmin import nmf
from proxmin import operators


_kwargs_proxmin_default = {
    'W': None,  # optional weights
    'prox_A': operators.prox_plus,
    'prox_S': operators.prox_plus,
    'proxs_g': None,
    'steps_g': None,
    'Ls': None,
    'slack': 0.9,
    'update_order': None,
    'steps_g_update': 'steps_f',
    'max_iter': 1000,
    'e_rel': 1e-3,
    'e_abs': 0,
    'traceback': None,
    'custom_prox_likelihood': None,
}


class BlockSDMM(object):

    def __init__(self, Y, A0, S0, kwargs_proxmin=None):
        self.Y_matrix = Y
        self.A_matrix_init = A0
        self.S_matrix_init = S0
        if kwargs_proxmin is None:
            kwargs_proxmin = _kwargs_proxmin_default
        self.kwargs_proxmin = kwargs_proxmin

    def optimize(self):
        A, S, hist = nmf.nmf_with_prox_f(self.Y_matrix, 
                                         self.A_matrix_init, 
                                         self.S_matrix_init, 
                                         **self.kwargs_proxmin)
        return A, S, hist
