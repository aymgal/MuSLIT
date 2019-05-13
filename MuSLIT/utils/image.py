__author__ = 'aymgal'

import numpy as np


def square_grid(n, step=1):
    a = np.arange(0, n, step=step)
    mesh = np.dstack(np.meshgrid(a, a)).reshape(-1, 2).astype(float)
    x = mesh[:, 0]
    y = mesh[:, 1]
    return x, y

def array_to_image(one_d_array):
    n2 = np.size(one_d_array)
    n  = int(np.sqrt(n2))
    two_d_shape = (n, n)
    try:
        two_d_array = one_d_array.reshape(two_d_shape)
    except ValueError as e:
        raise ValueError("Image needs to be defined on square grid !"+
                         "\nOriginal error : {}".format(e))
    return two_d_array
    
def image_to_array(two_d_array):
    n = two_d_array.shape[0]
    one_d_shape = (n**2,)
    return two_d_array.reshape(one_d_shape)
    
def array_to_multiband(two_d_array):
    num_bands, n2 = two_d_array.shape
    n = int(np.sqrt(n2))
    three_d_shape = (num_bands, n, n)
    return two_d_array.reshape(three_d_shape)
    
def multiband_to_array(three_d_array):
    num_bands, n, _ = three_d_array.shape
    two_d_shape = (num_bands, n**2)
    return three_d_array.reshape(two_d_shape)
    