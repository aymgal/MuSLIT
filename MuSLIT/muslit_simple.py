__author__ = 'aymgal'

import numpy as np
from scipy import signal as scp
import functools

import MuSLIT.operators.all as all_ops
import MuSLIT.utils.image as image_utils
import MuSLIT.utils.math as math_utils
from MuSLIT.optimizers.condatvu import CondatVuOptimizer
from MuSLIT.transforms.starlet import StarletTransform
from MuSLIT.transforms.data_structures import ComponentMatrix


_mode = 'analysis'


class LightModellerSimple(object):

    """TODO : check matrix shapes all over the place"""

    def __init__(self, target_image, forward_model, starlet_level=0, 
                 source_to_image_ratio=1, threshold=1):

        self.mode = _mode
        self.target_image  = target_image
        self.forward_model = forward_model

        _, self.num_components = mixing_matrix.shape
        self.A = mixing_matrix

        self.num_bands, num_pix1, num_pix2 = target_image.shape
        if num_pix1 != num_pix2:
            raise ValueError("Input image must be square")

        self.num_pix = num_pix1
        self.num_pix_src = int(self.num_pix * source_to_image_ratio)
        n2  = self.num_pix**2
        ns2 = self.num_pix_src**2

        self.starlet = StarletTransform(lvl=starlet_level)

        # TODO update threshold schedule
        self.thresh_schedule: lambda i: return 5. * np.exp(-i)

        self._init_optimizer()


    def run(self, n_iter=100):
        self._random_init()
        self.optimizer(self.X0, self.U0, n_iter=n_iter)


    def _init_optimizer(self):
        self.forward_op   = self.forward_model.operator
        self.forward_op_t = self.forward_model.transpose
        self.transform_op   = self.starlet.analysis
        self.transform_op_t = self.starlet.synthesis  # APPOXIMATION !!!

        Lip = self.forward_model.lipschitz
        tau = 1. / Lip
        eta = 0.5 * Lip

        prox1 = lambda X, s, t: all_ops.prox_soft_thresh(X, s, thresh=t)
        prox2 = lambda X, s, t: all_ops.prox_plus(X, s)
    
        self.optimizer = CondatVuOptimizer(self.target_image, 
                                           self.forward_op, self.forward_op_t, 
                                           self.transform_op, self.transform_op_t,
                                           self.thresh_schedule,
                                           prox1, prox2, n_iter, tau, nu)

    def _random_init(self):
        U0_l = LightComponent(self.num_pix, random_init=True)
        U0_s = LightComponent(self.num_pix_src, random_init=True)
        self.U0 = ComponentMatrix([U0_l.data, U0_s.data])
        
        X0_l = LightComponent(self.num_pix, random_init=True)
        X0_s = LightComponent(self.num_pix_src, random_init=True)
        self.X0 = ComponentMatrix([X0_l.data, X0_s.data])





