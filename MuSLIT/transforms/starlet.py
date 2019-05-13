__author__ = 'aymgal'

import numpy as np
import pysap


class BaseWaveletTransform(object):

    def __init__(self, lvl):
        self.nb_scale = lvl + 1  # + 1 for the coarsest scale


class SterletTransform(BaseWaveletTransform):

    _transform_name = 'BsplineWaveletTransformATrousAlgorithm'

    def __init__(self, lvl=1, n_omp_threads=None):
        super().__init__(lvl)
        transf_kwargs = {}
        if n_omp_threads is not None:
            transf_kwargs['nb_procs'] = n_omp_threads

        # note that if 'n_omp_threads' is not provided, 
        # PySAP will automatically set it the 
        # max number of CPUs available minus 1
        pysap_tranform = pysap.load_transform(_transform_name)
        self._T = pysap_tranform(nb_scale=self.nb_scale, 
                                 verbose=False, **transf_kwargs)

    def transform(self, img):
        """Undecimated wavelet transform with Bspline wavelet,
        i.e. 1st gen starlet transform using PySAP.
        """
        # set the image
        self._T.data = img
        self._T.analysis()
        coeffs = self._T.analysis_data
        return coeffs

    def inverse(coeffs, fast=True):
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
