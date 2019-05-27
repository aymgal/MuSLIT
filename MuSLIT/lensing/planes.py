__author__ = 'aymgal'


import numpy as np
import scipy.signal as scp

import MuSLIT.utils.coordinates as coord_utils


def kappa_to_alpha(kappa, num_pix):
    #Computes the deflection angle of a single photon at coordinates theta in the source plane and a lens
    #mass distribution kappa

    num_pix_kappa = kappa.shape[0]

    #Coordonnees de la grille de l'espace image
    x, y = coord_utils.square_grid(num_pix_kappa, dtype=float)
    x = x.reshape((num_pix_kappa, num_pix_kappa))
    y = y.reshape((num_pix_kappa, num_pix_kappa))

    x0 = num_pix_kappa / 2.
    y0 = num_pix_kappa / 2.

    xs = x - x0
    ys = y - y0

    r2 = xs**2 + ys**2
    l0 = np.where(r2 == 0)

    tabx = xs / r2
    taby = ys / r2
    tabx[l0] = 0.
    taby[l0] = 0.

    # perform the integration from the formula kappa -> alpha
    inte_x = scp.fftconvolve(tabx, kappa, mode='same') / np.pi
    inte_y = scp.fftconvolve(taby, kappa, mode='same') / np.pi

    # select the central part of the larger grid
    alpha_x = inte_x[int(x0-num_pix/2):int(x0+num_pix/2),
                     int(y0-num_pix/2):int(y0+num_pix/2)]

    alpha_y = inte_y[int(x0-num_pix/2):int(x0+num_pix/2),
                     int(y0-num_pix/2):int(y0+num_pix/2)]

    return alpha_x, alpha_y


def build_lensing_operator(kappa, num_pix, source_to_image_ratio, 
                           x_shear=0, y_shear=0, alpha_x_in=None, alpha_y_in=None):
    """theta positions for each pixel in beta"""

    if (alpha_x_in is not None) and (alpha_y_in is not None):
        print("Deflection angles have been provided")
        alpha_x = alpha_x_in
        alpha_y = alpha_y_in
    else:
        alpha_x, alpha_y = kappa_to_alpha(kappa, num_pix)

    alpha_x = alpha_x + x_shear
    alpha_y = alpha_y + y_shear

    xa, ya = coord_utils.square_grid(num_pix, dtype=int)
    xa = xa.reshape((num_pix, num_pix))
    ya = ya.reshape((num_pix, num_pix))

    num_pix_src = int(num_pix * source_to_image_ratio)
    xb, yb = coord_utils.square_grid(num_pix_src, dtype=int)

    #Scaling of the source grid

    #Scaling of the deflection grid
    xa = xa * float(num_pix_src) / float(num_pix)
    ya = ya * float(num_pix_src) / float(num_pix)
    
    F = []
    for i in range(np.size(xb)):
        #Deflection of photons emitted in xb[i],yb[i]
        theta_x = xb[i] * float(num_pix_src) / float(num_pix_src) + alpha_x
        theta_y = yb[i] * float(num_pix_src) / float(num_pix_src) + alpha_y

        #Matching of arrivals with pixels in image plane
        xprox = np.int_(np.abs((xa - theta_x) * 2))
        yprox = np.int_(np.abs((ya - theta_y) * 2))

        if np.min(xprox + yprox) == 0:
            loc2 = np.array(np.where((xprox + yprox) == 0)) * float(num_pix) / float(num_pix)
        else:
            loc2 = []

        if np.size(loc2) == 0:
            F.append([0])
        else:
            F.append(np.int_(loc2))

    return F


def source_to_image(source, lensing_operator, num_pix, ones=True):
    if ones:
        one_lens = source_to_image(np.ones(source.shape), lensing_operator, num_pix, ones=False)
        one_lens[np.where(one_lens == 0)] = 1.
    else:
        one_lens = 1.

    image = np.zeros((num_pix, num_pix))

    num_pix_src = source.shape[0]
    xb, yb = coord_utils.square_grid(num_pix_src, dtype=int)

    k = 0
    for pos in lensing_operator:
        if np.size(np.shape(pos)) != 1:
            image[np.array(pos[0][:]),
                  np.array(pos[1][:])] += source[yb[k], xb[k]]
        k += 1
    return image / one_lens


def image_to_source(image, lensing_operator, num_pix_src, square=False, lensed=True):
    source = np.zeros((num_pix_src, num_pix_src))
    xb, yb = coord_utils.square_grid(num_pix_src, dtype=int)

    for k in range(num_pix_src**2):
        pos = lensing_operator[k]
        if np.size(np.shape(pos)) > 1:

            light = image[np.array(pos[0][:]), np.array(pos[1][:])]

            if lensed:
                light /= np.max([1, np.size(pos[0][:])])
                if square:
                    source[yb[k], xb[k]] += np.sum(light**2)
                else:
                    source[yb[k], xb[k]] += np.sum(light)
            else:
                source[yb[k], xb[k]] += np.sum(light)
    if square:
        source = np.sqrt(source)
    return source
