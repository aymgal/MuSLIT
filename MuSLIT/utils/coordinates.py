__author__ = 'aymgal'

import numpy as np


def square_grid(num_pix, delta_pix=1, step=1, dtype=float):
    a = np.arange(0, num_pix, step=step) * delta_pix
    mesh = np.dstack(np.meshgrid(a, a)).reshape(-1, 2).astype(dtype)
    x = mesh[:, 0]
    y = mesh[:, 1]
    return x, y