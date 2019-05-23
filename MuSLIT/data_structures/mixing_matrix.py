__author__ = 'aymgal'


import numpy as np

# from MuSLIT.data_structures.mb_image import MultibandImage
from MuSCADeT.MCA import PCA_initialise



def perform_PCA(img):
    return PCA_initialise(img, self.Nc, angle=40, npca=32,
                          alpha=[0, 0], plot=0, newwave=0)


class MixingMatrix(object):
    """Base class that describes the A matrix in the model Y = H * A * X * F + N
    """

    def __init__(self, num_bands, num_components, random_init=True, input_image=None):
        self.Nb = num_bands
        self.Nc = num_components
        self.data_shape = (self.Nb, self.Nc)
        if random_init:
            self.data = np.random.rand(self.data_shape)
            # normalize columns to sum to 1
            for i in range(self.Nc):
                self.data[:, i] = self.data[:, i]/self.data[:, i].sum()
        else:
            if input_image is None:
                raise ValueError("Input image required for PCA initialization")
            self.data = perform_PCA(input_image)

    def A(self):
        return self.apply_as_operator

    def A_t(self):
        return self.apply_as_transpose_operator

    def apply_as_operator(self, component_matrix):
        return self._apply(component_matrix)

    def apply_as_transpose_operator(self, component_matrix):
        return self._apply_transpose(component_matrix)

    def _apply(self, cmo):
        return self.data * cmo.data

    def _apply_transpose(self, cmo):
        return cmo.data * self.data.T
