__author__ = 'aymgal'


import numpy as np

from pylensmodels.mass.spemd import SPEMD_glee
from pylensmodels.light.gaussian import Gaussian, GaussianElliptical
from pylensmodels.utils import coordinates
from MuSLIT.utils import image
from MuSLIT.lensing import planes


lin = lambda x: x
log = lambda x: np.log10(x)



def simple_multiband_lens(sigma_noise=0.05):

    num_pix = 100
    x_grid, y_grid = coordinates.square_grid(num_pix)
    # print(x_grid, y_grid)


    source_to_image_ratio = 1
    num_pix_src = num_pix * source_to_image_ratio

    kwargs_spemd = {
        'x0': 50.5,  # pixels (origin : lower left pixel)
        'y0': 50.5,  # pixels (origin : lower left pixel)
        'gamma': 0.3,
        'theta_E': 25.,  # pixels
        'q': 0.8,
        'phi': 0.3,
        'r_core': 0.01,  # pixels
    }

    # WARNING : no Dds/Ds (physical to scaled conversion) scaling
    mass_model = SPEMD_glee(kwargs_spemd, Dds_Ds=None)
    # kappa = mass_model.convergence(x_grid, y_grid)
    alpha1, alpha2 = mass_model.deflection(x_grid, y_grid)


    lensing_operator = planes.build_lensing_operator(None, num_pix, source_to_image_ratio, 
                                                     alpha_x_in=alpha1, alpha_y_in=alpha2)
    # print(type(lensing_operator), len(lensing_operator))


    kwargs_gaussian_ell = {'x0': 51, 'y0': 52, 'sigma': 2, 'phi': 0.3, 'q': 0.6, 'amp': 1}
    gaussian_source = GaussianElliptical(kwargs_gaussian_ell).function(x_grid, y_grid)

    kwargs_gaussian = {'x0': 50, 'y0': 50, 'sigma_x': 8, 'sigma_y': 8, 'amp': 7}
    gaussian_lens = Gaussian(kwargs_gaussian).function(x_grid, y_grid)


    # normalization
    galaxy_source = gaussian_source/gaussian_source.max()
    galaxy_lens   = gaussian_lens/gaussian_lens.max()


    galaxy_lensed = planes.source_to_image(galaxy_source, lensing_operator, num_pix)
    # print(galaxy_lensed.max())

    galaxy_unlensed = planes.image_to_source(galaxy_lensed, lensing_operator, num_pix_src)
    # print(galaxy_unlensed.max())


    sim_lens_base = galaxy_lensed + galaxy_lens


    lens_SEDs   = (1.0, 0.7, 0.4)   # not normalized
    source_SEDs = (0.6, 0.7, 0.8)   # not normalized

    lens_multiband = [galaxy_lens * sed for sed in lens_SEDs]
    lens_multiband = image.bands_to_image(lens_multiband)

    lensed_multiband = [planes.source_to_image(galaxy_source*sed, lensing_operator, num_pix) for sed in source_SEDs]
    lensed_multiband = image.bands_to_image(lensed_multiband)


    sim_lens_multiband_no_noise = lens_multiband + lensed_multiband


    noise1 = np.random.randn(num_pix, num_pix) * sigma_noise  # arbitrary units for now
    noise2 = np.random.randn(num_pix, num_pix) * sigma_noise
    noise3 = np.random.randn(num_pix, num_pix) * sigma_noise
    noise_multiband = image.bands_to_image([noise1, noise2, noise3])

    sim_lens_multiband = sim_lens_multiband_no_noise + noise_multiband

    return sim_lens_multiband
