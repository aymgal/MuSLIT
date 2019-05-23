__author__ = 'aymgal'

import numpy as np


class BaseDWT(object):

    def __init__(self, lvl, filter_length=5):
        # the number of scale equals the number decomposition levels + 1 for coarsest scale
        self.nb_scale = lvl + 1
        self.filter_length = filter_length
        self._pysap_transform = None
            
    def _check_lvl(self, img):
        max_lvl = BaseDWT.max_wavelet_lvl_ND(img.shape, self.filter_length)
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
