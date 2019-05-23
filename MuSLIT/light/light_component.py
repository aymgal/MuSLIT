__author__ = 'aymgal'


import numpy as np

import MuSLIT.utils.image as image_utils


class LightComponent(object):
    """Base class that describes a light component/source in the image.
    Typically either the lens galaxy G or the source galaxy S.
    """

    def __init__(self, num_pix, random_init=True, init_data=None):
        self.Np = num_pix
        self._update_shapes()
        if random_init:
            self._data = np.random.rand(self.data_shape)
        elif init_data is not None:
            self._data = init_data
        else:
            self._data = np.zeros(self.data_shape)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        if new_data.shape != self.data_shape:
            Np = np.sqrt(new_shape)
            if not Np.is_integer():
                raise RuntimeError("Inconsistent new data size")
            self.Np = int(Np)
            self._update_shapes()

    @staticmethod
    def light(data):
        return image_utils.array_to_image(data)

    def _update_shapes(self):
        self.data_shape = (self.Np**2,)
        self.img_shape  = (self.Np, self.Np)
