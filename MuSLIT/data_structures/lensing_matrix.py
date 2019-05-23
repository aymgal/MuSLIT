__author__ = 'aymgal'


import numpy as np

from MuSLIT.lensing.planes import build_lensing_operator


def lensing_op(deflection_map, np, stir):
    alpha_x, alpha_y = deflection_map
    return planes.build_lensing_operator(None, np, stir, alpha_x_in=alpha_x, alpha_y_in=alpha_y):


class LensingMatrix(object):
    """Base class that describes the A matrix in the model Y = H * A * X * F + N
    """

    def __init__(self, deflection_map, num_pix, source_to_image_ratio):
        self.num_pix =  num_pix
        self.num_pix_src = int(num_pix * source_to_image_ratio)
        self.lensing = lensing_op(deflection_map, num_pix, source_to_image_ratio)

    def source_to_image(self, source):
        return planes.source_to_image(source, self.lensing, self.num_pix)

    def image_to_source(self, image, square=False):
        return planes.image_to_source(image, self.lensing, 
                                      self.num_pix_src, square=square)

    def F(self):
        return self.apply_as_operator

    def apply_as_operator(self, component_matrix):
        return self._apply(component_matrix)

    def _apply(self, X):
        