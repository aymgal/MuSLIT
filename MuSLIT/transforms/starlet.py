__author__ = 'aymgal'

import numpy as np
import pysap


_starlet_pysap_name = 'BsplineWaveletTransformATrousAlgorithm'


class BaseDWT(object):

    def __init__(self, lvl, filter_length=5):
        # the number of scale equals the number decomposition levels + 1 for coarsest scale
        self.nb_scale = lvl + 1
        self.filter_length = filter_length
            
    def _check_lvl(self, img):
        max_lvl = BaseDWT.max_wavelet_lvl_ND(img.shape, self.filter_length)
        if (self.nb_scale) < 2 or (self.nb_scale > max_lvl + 1):
            self.nb_scale = max_lvl + 1
        print("Wavelet number of scales set to {}".format(self.nb_scale))

    @staticmethod
    def max_wavelet_lvl_1D(signal_length, filter_length):
        #max_lvl = np.log2( signal_length / (filter_length - 1) )  # from PyWavelets...but seems wrong
        max_lvl = np.log2(signal_length)
        return int(np.floor(max_lvl))
        
    @staticmethod
    def max_wavelet_lvl_ND(signal_shape, filter_length):
        max_lvl_list = []
        for axis_length in signal_shape:
            max_lvl_list.append(BaseDWT.max_wavelet_lvl_1D(axis_length, filter_length))
        return min(max_lvl_list)


class StarletTransform(BaseDWT):

    def __init__(self, lvl=0, n_omp_threads=None):
        super().__init__(lvl, filter_length=5)
        self.transf_kwargs = {}
        if n_omp_threads is not None:
            self.transf_kwargs['nb_procs'] = n_omp_threads

        # note that if 'n_omp_threads' is not provided, 
        # PySAP will automatically set it the 
        # max number of CPUs available minus 1
        self._pysap_tranform = pysap.load_transform(_starlet_pysap_name)
        
    def transform(self, img):
        """Undecimated wavelet transform with Bspline wavelet,
        i.e. 1st gen starlet transform using PySAP.
        """
        self._check_lvl(img)  # update decomposition level if inconsistent
        self._T = self._pysap_tranform(nb_scale=self.nb_scale, 
                                       verbose=False, **self.transf_kwargs)
        # set the image
        self._T.data = img
        self._T.analysis()
        coeffs = self._T.analysis_data
        return coeffs

    def inverse(self, coeffs, fast=True):
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
