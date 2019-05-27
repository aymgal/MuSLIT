__author__ = 'aymgal'

import numpy as np


def array2pysap(a):
    a_list = []
    for i in range(a.shape[0]):
        a_list.append(a[i, :, :])
    return a_list

def pysap2array(a):
    return np.asarray(a)


class BaseDWT(object):

    def __init__(self, img_shape, lvl=0, filter_length=5, filter_name=None, 
                 n_omp_threads=None):
        self.filter_name = filter_name
        self.filter_length = filter_length

        # the number of scale equals the number decomposition levels + 1 for coarsest scale
        self.nb_scale = lvl + 1
        self._check_lvl(img_shape)

        self.transf_kwargs = {}
        if n_omp_threads is not None:
            self.transf_kwargs['nb_procs'] = n_omp_threads

        # note that if 'n_omp_threads' is not provided, 
        # PySAP will automatically set it the 
        # max number of CPUs available minus 1

        if self.filter_name is not None:
            self._pysap_transform = pysap.load_transform(self.filter_name)
        else:
            self._pysap_transform = None

        self._update_tranform()

    def analysis(self, img):
        """Analysis operator
        """
        self._check_lvl(img.shape)  # update decomposition level if inconsistent

        # set the image
        self._T.data = img
        self._T.analysis()
        coeffs = self._T.analysis_data
        return pysap2array(coeffs)

    def synthesis(self, coeffs):
        """Synthesis operator
        """
        self._T.analysis_data = array2pysap(coeffs)
        image = self._T.synthesis()
        recon = image.data
        return recon
            
    def _check_lvl(self, img_shape):
        max_lvl = BaseDWT.max_wavelet_lvl_ND(img_shape, self.filter_length)
        if (self.nb_scale) < 2 or (self.nb_scale > max_lvl + 1):
            self.nb_scale = max_lvl + 1
            self._update_tranform()
        print("Wavelet number of scales set to {}".format(self.nb_scale))

    def _update_tranform(self):
        if self._pysap_transform is not None:
            self._T = self._pysap_transform(nb_scale=self.nb_scale, 
                                            verbose=False, **self.transf_kwargs)

    @staticmethod
    def max_wavelet_lvl_1D(signal_length, filter_length):
        max_lvl = np.floor(np.log2(signal_length))
        return int(max_lvl)
        
    @staticmethod
    def max_wavelet_lvl_ND(signal_shape, filter_length):
        max_lvl_list = []
        for axis_length in signal_shape:
            max_lvl_list.append(BaseDWT.max_wavelet_lvl_1D(axis_length, filter_length))
        return min(max_lvl_list)
