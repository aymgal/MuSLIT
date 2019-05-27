__author__ = 'aymgal'


import numpy as np

import MuSLIT.utils.math as math_utils


class BlurringMatrix(object):
    """Class that describes a multiband blurring 'operator' H in the model Y = H * A * X * F + N
    """

    def __init__(self, psf_list, psf_conj_list=None):
        self.Nb = len(psf_list)
        self._psf_list = psf_list
        if psf_conj_list is not None:
            self._psf_conj_list = psf_conj_list
        else:
            self._psf_conj_list = [math_utils.conjugate(psf) for psf in psf_list]

    def H(self):
        return self.apply_as_operator

    def H_t(self):
        return self.apply_as_transpose_operator

    def apply_as_operator(self, multiband_image):
        return self._apply(multiband_image, self._psf_list)

    def apply_as_transpose_operator(self, multiband_image):
        return self._apply(multiband_image, self._psf_conj_list)

    def _apply(self, mb_img, psf_list):
        for i in range(self.Nb):
            psf_i = psf_list[i]
            img_i = mb_img[i, :, :]
            mb_img[i, :, :] = math_utils.convolve(img_i, psf_i)
        return mb_img
