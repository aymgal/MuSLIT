__author__ = 'aymgal'


import numpy as np

import MuSLIT.utils.image as image_utils
import MuSLIT.utils.noise as noise_utils


class MultibandImage(object):
    """Class that describes the input multiband image Y.
    """

    def __init__(self, image_list):
        multiband = np.asarray(image_list)
        self._data = image_utils.multiband_to_array(multiband)
        self._update_properties()

    @property
    def num_bands(self):
        return self.Nb

    @property
    def num_pix(self):
        return self.Np

    @property
    def image(self):
        return image_utils.array_to_multiband(self.data)

    @image.setter
    def image(self, new_image):
        new_data = image_utils.multiband_to_array(new_image)
        self.data(new_data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._update_properties()

    @property
    def noise_MAD(self):
        if not hasattr(self, '_noise_mad'):
            self._noise_mad = multiband_MAD_estimator(self)
        return self._noise_mad

    def _update_properties(self):
        def self._noise_mad
        self.Nb, Np2 = self.data.shape
        Np = np.sqrt(Np2)
        if not Np.is_integer():
            raise RuntimeError("Inconsistent new data size")
        else:
            self.Np = int(Np)

    @staticmethod
    def multiband_MAD_estimator(mb_img):
        mb_noise = []
        for i in range(mb_img.num_bands):
            img = mb_img.image[i, :, :]
            sigma = noise_utils.compute_MAD_estimtor(img)
            mb_noise.append(sigma)
        return mb_noise
