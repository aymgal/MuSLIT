__author__ = 'aymgal'


import numpy as np


class ComponentMatrix(object):
    """Class that describes the X matrix in the model Y = H * A * X * F + N
    """

    def __init__(self, component_list):
        self.Ns = len(component_list)
        self._component_list = component_list
        self._update_sizes()

    def __add__(self, other):
        """addition of two ComponentMatrix"""
        lens_light_data = self.lens_data + other.lens_data
        source_light_data = self.source_data + other.source_data
        return ComponentMatrix([lens_light_data, source_light_data])

    def __sub__(self, other):
        """subtraction of two ComponentMatrix"""
        lens_light_data = self.lens_data - other.lens_data
        source_light_data = self.source_data - other.source_data
        return ComponentMatrix([lens_light_data, source_light_data])

    def __rmul__(self, other):
        """reflected multiplication, when 'other' is not a ComponentMatrix"""
        if isinstance(other, (int, float, long)):
            lens_light_data = other * self.lens_data
            source_light_data = other * self.source_data
        elif isinstance(other, (tuple, list)) or \
                (isinstance(other, np.ndarray) and other.ndim == 1):
            lens_light_data = other[0] * self.lens_data
            source_light_data = other[1] * self.source_data
        else:
            raise NotImplementedError("Unsupported operation")
        return ComponentMatrix([lens_light_data, source_light_data])

    @property
    def lens_data(self):
        return self._component_list[0]

    @lens_data.setter
    def lens_data(self, data):
        self._component_list[0] = data
        self._update_sizes()

    @property
    def source_data(self):
        return self._component_list[1]

    @source_data.setter
    def source_data(self, data):
        self._component_list[1] = data
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

    def norm(self, p=2):
        """Average of the norm among components"""
        norm_lens   = np.linalg.norm(self.lens_data, p)
        norm_source = np.linalg.norm(self.source_data, p)
        return (norm_lens + norm_source) / 2.


    def _update_size(self):
        lens_data_shape = self._component_list[0].img_shape
        if len(lens_data_shape) > 2:
            # means that it is in a transformed domain
            # the first dimension is the number of scales
            self.Np  = lens_data_shape[1]
        else:
            # the first dimension is the number side pixels
            self.Np  = lens_data_shape[0]

        source_data_shape = self._component_list[0].img_shape
        if len(source_data_shape) > 2:
            # means that it is in a transformed domain
            # the first dimension is the number of scales
            self.Nps  = lens_data_shape[1]
        else:
            # the first dimension is the number side pixels
            self.Nps  = lens_data_shape[0]
