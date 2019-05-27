__author__ = 'aymgal'

import numpy as np

from MuSLIT.transforms.base_dwt import BaseDWT
from MuSLIT.transforms.data_structures import ComponentMatrix



class Starlet1stGen(BaseDWT):
    """Undecimated wavelet transform with Bspline wavelet using PySAP"""

    def __init__(self, lvl=0, n_omp_threads=None):
        filter_name   = 'BsplineWaveletTransformATrousAlgorithm'
        filter_length = 5

        super().__init__(lvl=lvl, filter_name=filter_name,
                         filter_length=filter_length,
                         n_omp_threads=n_omp_threads)

    def synthesis(self, coeffs, fast=True):
        if fast:
            # for 1st gen starlet the reconstruction can be performed by summing all scales 
            return np.sum(coeffs, axis=0)
        else:
            # use set the analysis coefficients
            super().synthesis(coeffs)
