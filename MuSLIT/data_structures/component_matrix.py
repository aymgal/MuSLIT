__author__ = 'aymgal'


import copy
import numpy as np


class ComponentMatrix(object):
    """Class that describes the X matrix in the model Y = H * A * X * F + N
    """

    def __init__(self, component_list):
        self.Ns = len(component_list)
        self._component_list = component_list
        self._update_sizes()

    def __add__(self, other):
        lens_light_data = self.lens_data + other.lens_data
        source_light_data = self.source_data + other.source_data
        return ComponentMatrix([lens_light_data, source_light_data])

    def __sub__(self, other):
        lens_light_data = self.lens_data - other.lens_data
        source_light_data = self.source_data - other.source_data
        return ComponentMatrix([lens_light_data, source_light_data])

    def __rmul__(self, other):
        """reflected multiplication, when 'other' is a number"""
        if isinstance(other, (int, float, long)):
            lens_light_data = other * self.lens_data
            source_light_data = other * self.source_data
        else:
            raise RuntimeError("Unsupported operation")
        return ComponentMatrix([lens_light_data, source_light_data])

    def __rdiv__(self, other):
        """reflected division, when 'other' is a number"""
        if isinstance(other, (int, float, long)):
            lens_light_data = other / self.lens_data
            source_light_data = other / self.source_data
        else:
            raise RuntimeError("Unsupported operation")
        return ComponentMatrix([lens_light_data, source_light_data])

    def _update_sizes(self):
        self.Nb, self.Np, _ = self._component_list[0].img_shape
        _, self.Nps, _ = self._component_list[1].img_shape

    @property
    def lens_data(self):
        return self._component_list[0].data

    @lens_data.setter
    def lens_data(self, data):
        self._component_list[0].data = data
        self._update_sizes()

    @property
    def source_data(self):
        return self._component_list[1].data

    @source_data.setter
    def source_data(self, data):
        self._component_list[1].data = data
        self._update_sizes()

    @property
    def data(self):
        if self.Np == self.Nps:
            return np.array([self.lens_data, self.source_data])
        else:
            raise RuntimeError("Cannot express ComponentMatrix as an array")

    @property
    def XF(self):
        return self.data
