__author__ = 'aymgal'


import numpy as np


class DictionnarySet(object):
    """Class that describes a concatenation of dictionnaries Phi.
    """

    def __init__(self, transform_list):
        self._transform_list = transform_list

    def Phi_t(self):
        return self.apply_as_operator

    def Phi(self):
        return self.apply_as_transpose_operator

    def analysis(self):
        return self.apply_as_operator
 
    def synthesis(self):
        return self.apply_as_transpose_operator

    def apply_as_transpose_operator(self, component_matrix):
        return self._apply_t(component_matrix)

    def apply_as_operator(self, component_matrix):
        return self._apply(component_matrix)

    def apply_as_transpose_operator(self, component_matrix):
        return self._apply_t(component_matrix)

    def _apply(self, cm):
        lens_transf = self._transform_list[0]
        cm.lens_data = lens_transf.analysis(cm.lens_data)
        source_transf = self._transform_list[1]
        cm.source_data = source_transf.analysis(cm.source_data)
        return cm

    def _apply_t(self, cm):
        lens_transf = self._transform_list[0]
        cm.lens_data = lens_transf.synthesis(cm.lens_data)
        source_transf = self._transform_list[1]
        cm.source_data = source_transf.synthesis(cm.source_data)
        return cm
        