__author__ = 'aymgal'


import numpy as np

from MuSLIT.lensing.planes import build_lensing_operator


def lensing_op(deflection_map, np, stir):



class LensingMatrix(object):
    """Base class that describes the A matrix in the model Y = H * A * X * F + N
    """

    def __init__(self, deflection_map, num_pix, source_to_image_ratio=1):
        self.num_pix =  num_pix
        self.num_pix_src = int(num_pix * source_to_image_ratio)
        
        alpha_x, alpha_y = deflection_map
        self._lensing_op = planes.build_lensing_operator(None, num_pix, 
                                                         source_to_image_ratio, 
                                                         alpha_x_in=alpha_x, 
                                                         alpha_y_in=alpha_y):

    def F(self):
        return self.apply_as_operator

    def F_inv(self):
        return self.apply_as_inverse_operator

    def apply_as_operator(self, component_matrix):
        return self._apply(component_matrix)

    def apply_as_inverse_operator(self, component_matrix):
        return self._apply_inverse(component_matrix)

    def _source_to_image(self, source):
        return planes.source_to_image(source, self._lensing_op, self.num_pix)

    def _image_to_source(self, image):
        return planes.image_to_source(image, self._lensing_op, self.num_pix_src)

    def _apply(self, cm):
        cm.source_data = self._source_to_image(cm.source_data)
        return cm

    def _apply_inverse(self, cm):
        cm.source_data = self._image_to_source(cm.source_data)
        return cm
