__author__ = 'aymgal'

import numpy as np
from scipy import signal as scp
import functools

import MuSLIT.operators.all as all_ops
import MuSLIT.utils.image as image_utils
import MuSLIT.utils.math as math_utils
from MuSLIT.lensing import planes
from MuSLIT.optimizers.condatvu import CondatVuOptimizer
from MuSLIT.transforms.starlet import StarletTransform


_mode = 'analysis'


class LightModellerSimple(object):

    """TODO : check matrix shapes all over the place"""

    def __init__(self, target_image, mixing_matrix, lensing_operator, 
                 bluring_operators, starlet_level=0, 
                 source_to_image_ratio=1, threshold=1):

        self.mode = _mode

        self.Y  = image_utils.multiband_to_array(target_image)

        _, self.num_components = mixing_matrix.shape
        self.A = mixing_matrix

        self.num_bands, num_pix1, num_pix2 = target_image.shape
        if num_pix1 != num_pix2:
            raise ValueError("Input image must be square")

        self.num_pix = num_pix1
        self.num_pix_src = int(self.num_pix * source_to_image_ratio)
        n2  = self.num_pix**2
        ns2 = self.num_pix_src**2

        self.psf_list = bluring_operator_list
        self.lensing_op = lensing_operator

        self.starlet = StarletTransform(lvl=starlet_level)
        self.thresh = threshold

        self._initialize_images()
        self._initialize_operators()
        self._initialize_optimizers()


    def _initialize_images(self):
        self.G0 = np.random.rand((n2,))
        self.S0 = np.random.rand((ns2,))


    def _initialize_operators(self):
        self.F     = self.source_to_image
        self.F_inv = self.image_to_source

        psf_conj_list = [math_utils.conjugate(psf) for psf in self.psf_list]
        self.H   = lambda X: self._apply_blurring(X, self.psf_list)
        self.H_t = lambda X: self._apply_blurring(X, self.psf_conj_list)

        self.A = lambda X: self.mixing_matrix * X
        self.A_t = lambda X: X * self.mixing_matrix.T

        # A_G = lambda X: self.A[:, 0] * X
        # self.forward_op_G   = lambda X: H( A_G( F(X) ) )
        # self.forward_op_G_t = lambda X: F_inv( A_G( H_t(X) ) )

        # A_S = lambda X: self.A[:, 1] * X
        # self.forward_op_S   = lambda X: H( A_G( F(X) ) )
        # self.forward_op_S_t = lambda X: F_inv( A_G( H_t(X) ) )


    def forward_op(self, X):
        X_ = np.copy(X)
        X_[1, :] = self.F(X[1, :])
        return self.H(self.A(X_))

    def forward_op_t(self, HAX):
        HAX_ = np.copy(HAX)
        X = self.A_t(self.H_t(HAX_))
        X[1, :] = self.F_inv(X[1, :])
        return X

    def transform_op(self, X):
        alpha = np.zeros((self.num_bands))

    def _apply_blurring(self, X, H_list):
        # loop over bands
        for i in range(X.shape[0]):
            X[i, :] = math_utils.convolve(H_list[i])
        return X

    def _initialize_optimizers(self):
        shape = (self.num_components, self.n2)
        Lip = math_utils.power_method_op(self.forward_op, self.forward_op_t, shape)

        tau = 1. / Lip
        eta = 0.5 * Lip

        transform_op   = self.starlet.tranform
        transform_op_t = self.starlet.inverse  # APPOXIMATION !!!
        prox1 = lambda X, step: all_ops.prox_soft_thresh(X, step, thresh=self.thresh)
        prox2 = lambda X, step: all_ops.prox_plus(X, step)
        self.optimizer_G = CondatVuOptimizer(self.forward_op, self.forward_op_t, 
                                             transform_op, transform_op_t,
                                             prox1, prox2, n_iter, tau, nu)

    def run(self, n_iter=100):


    def source_to_image(self, source):
        return planes.source_to_image(source, self.lensing_op, self.num_pix)


    def image_to_source(self, image, square=False):
        return planes.image_to_source(image, self.lensing_op, 
                                      self.num_pix_src, square=square)



