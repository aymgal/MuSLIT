# __author__ = 'aymgal'


# import numpy as np

# import MuSLIT.utils.image as image_utils


# class MultibandImage(object):
#     """Class that describes the input multiband image Y.
#     """

#     def __init__(self, image_list):
#         self.Nb = len(image_list)
#         self.Np = image_list[0].shape[0]
#         multiband = np.asarray(image_list)
#         self._data = image_utils.multiband_to_array(multiband)

#     def __add__(self, other):
        

#     @property
#     def image(self):
#         return image_utils.array_to_multiband(self.data)

#     @image.setter
#     def image(self, new_image):
#         new_data = image_utils.multiband_to_array(new_image)
#         self.data(new_data)

#     @property
#     def data(self):
#         return self._data

#     @data.setter
#     def data(self, new_data):
#         self.