__author__ = 'aymgal'

import numpy as np
from scipy.ndimage import filters


def compute_MAD_estimtor(img, filter_length=3):
    filter_shape = (filter_length, filter_length)
    meda = filters.median_filter(img, size=filter_shape)
    medfil = np.abs(x - meda) #np.median(x))
    sigma = 1.48 * np.median(medfil)
    return sigma

def compute_MOM_estimator(img):
    # TODO
    pass