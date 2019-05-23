__author__ = 'aymgal'

import numpy as np
import pysap

from MuSLIT.transforms.base_dwt import BaseDWT
from MuSLIT.transforms.data_structures import ComponentMatrix


_starlet_pysap_name = 'BsplineWaveletTransformATrousAlgorithm'
_starlet_filter_length = 5


class StarletTransform(BaseDWT):

    def __init__(self, lvl=0, n_omp_threads=None):
        super().__init__(lvl, filter_length=_starlet_filter_length)
        self.transf_kwargs = {}
        if n_omp_threads is not None:
            self.transf_kwargs['nb_procs'] = n_omp_threads

        # note that if 'n_omp_threads' is not provided, 
        # PySAP will automatically set it the 
        # max number of CPUs available minus 1
        self._pysap_transform = pysap.load_transform(_starlet_pysap_name)
        
    def analysis(self, img):
        """Undecimated wavelet transform with Bspline wavelet,
        i.e. 1st gen starlet transform using PySAP.
        """
        self._check_lvl(img)  # update decomposition level if inconsistent

        # set the image
        self._T.data = img
        self._T.analysis()
        coeffs = self._T.analysis_data
        return coeffs

    def synthesis(self, coeffs, fast=True):
        """Inverse 1st gen starlet transform
        """
        if fast:
            # for 1st gen starlet the reconstruction can be performed by summing all scales 
            recon = np.sum(coeffs, axis=0)
        else:
            # use set the analysis coefficients
            self._T.analysis_data = coeffs
            image = self._T.synthesis()
            recon = image.data
        return recon
