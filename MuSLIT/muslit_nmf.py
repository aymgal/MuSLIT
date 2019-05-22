__author__ = 'aymgal'

import numpy as np
import functools

import MuSLIT.operators.all as all_ops
import MuSLIT.utils.image as image_utils
from MuSLIT.lensing import planes
from MuSLIT.optimizers.bsdmm import BlockSDMM
from MuSLIT.transforms.starlet import StarletTransform


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def prox_likelihood(X, step, Xs=None, j=None, Y=None, WA=None, WS=None, 
                    prox_S=all_ops.prox_id, prox_A=all_ops.prox_id,
                    synthesis_op_S=None):
    if synthesis_op_S is not None:
        Xs_[1] = synthesis_op_S(Xs[1])
        if j == 1:
            X_ = synthesis_op_S(X[1])
    else:
        X_, Xs_= X, Xs

    import proxmin

    if j == 0:
        return proxmin.nmf.prox_likelihood_A(X_, step, S=Xs_[1], Y=Y, prox_g=prox_A, W=WA)
    else:
        return proxmin.nmf.prox_likelihood_S(X_, step, A=Xs_[0], Y=Y, prox_g=prox_S, W=WS)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


_num_components = 2
_mode = 'synthesis'


class LightModeller(object):

    """TODO : check matrix shapes all over the place"""

    def __init__(self, target_image, lensing_operator, threshold=0, 
                 starlet_level=1, source_to_image_ratio=1):
        self.mode = _mode
        self.num_components = _num_components

        self.num_bands, num_pix1, num_pix2 = target_image.shape
        assert num_pix1 == num_pix2, "Inout image must be square"

        self.num_pix = num_pix1
        self.num_pix_src = int(self.num_pix * source_to_image_ratio)
        n2  = self.num_pix**2
        ns2 = self.num_pix_src**2

        self.lensing_operator = lensing_operator

        self.F     = self.source_to_image
        self.F_inv = self.image_to_source

        self.Y  = image_utils.multiband_to_array(target_image)
        self.A0 = np.random.rand(self.num_bands, self.num_components)
        self.S0 = np.random.rand(self.num_components, n2)

        self.starlet = StarletTransform(lvl=starlet_level)

        self.thresh = threshold

        self._setup_optimizer()


    def run(self):
        # run the optimizer
        M1, M2, hist = self.optimizer.optimize()

        if self.mode == 'synthesis':
            S_t, G_t = M2[0, :], M2[1, :]
            S_t = image_utils.array_to_image(S_t)
            G_t = image_utils.array_to_image(G_t)
            S = self.starlet.inverse(S_t)
            G = self.starlet.inverse(G_t)

        elif self.mode == 'analysis':
            raise NotImplementedError("'Analysis' formulation not supported yet")

        return A, S, G, hist


    def synthesis_transform_S(self, X):
        # extract light components that are in wavelet domain
        S_t, G_t = X[0, :], X[1, :]

        # convert to 2D images
        S_t = image_utils.array_to_image(S_t)
        G_t = image_utils.array_to_image(G_t)
        print("SHAPES T", S_t.shape, G_t.shape)

        # apply the inverse starlet tranform
        S = self.starlet.inverse(S_t)
        G = self.starlet.inverse(G_t)
        print("SHAPES", S.shape, G.shape)

        # ray-trace to image plane to get FS
        FS = self.F(S)

        # convert to 1D array
        FS = image_utils.image_to_array(FS)
        G  = image_utils.image_to_array(G)

        X_new = np.zeros_like(X)  # TODO : really need a copy here ?!
        X_new[0, :] = FS
        X_new[1, :] = G
        
        return X_new


    # def analysis_prox_lensing_starlet_soft_thresh(self, X, step):
    #     """X is altered by this function !"""
    #     # extract light components
    #     FS, G = X[0, :], X[1, :]

    #     # convert to 2D images
    #     FS = image_utils.array_to_image(FS)
    #     G  = image_utils.array_to_image(G)

    #     # back ray-trace to source plane to get S
    #     S = self.F_inv(FS)

    #     # apply starlet transform
    #     S_t = self.starlet.tranform(S)
    #     G_t = self.starlet.tranform(G)

    #     # prox for l0 norm : soft-threshold coefficients
    #     S_t = all_ops.prox_soft_thresh(S_t, step, thresh=self.thresh)
    #     G_t = all_ops.prox_soft_thresh(G_t, step, thresh=self.thresh)

    #     # apply the inverse starlet tranform
    #     S = self.starlet.inverse(S_t)
    #     G = self.starlet.inverse(G_t)

    #     # ray-trace to image plane to get FS
    #     FS = self.F(S)

    #     # reshape to 1D arrays
    #     FS = image_utils.image_to_array(FS)
    #     G  = image_utils.image_to_array(G)

    #     # rebuild the matrix
    #     X[0, :] = FS
    #     X[1, :] = G
    #     return X


    def synthesis_prox_lensing_starlet_soft_thresh(self, X, step):
        """X is altered by this function !"""
        # prox for l1 norm : soft-threshold coefficients
        X = all_ops.prox_soft_thresh(X, step, thresh=self.thresh)
        return X


    def synthesis_prox_lensing_starlet_hard_thresh(self, X, step):
        """X is altered by this function !"""
        # prox for l0 norm : hard-threshold coefficients
        X = all_ops.prox_hard_thresh(X, step, thresh=self.thresh)
        return X


    def source_to_image(self, source):
        return planes.source_to_image(source, self.lensing_operator, self.num_pix)


    def image_to_source(self, image, square=False):
        return planes.image_to_source(image, self.lensing_operator, 
                                      self.num_pix_src, square=square)

    def _setup_optimizer(self):
        prox_A = all_ops.prox_plus
        prox_S = all_ops.prox_plus

        proxs_g = [
            # ops for A
            [
                all_ops.prox_column_norm,
            ],
            # ops for S
            [
                self.synthesis_prox_lensing_starlet_hard_thresh,
            ]
        ]

        modified_prox_likelihood = functools.partial(prox_likelihood, synthesis_op_S=self.synthesis_transform_S)

        kwargs_proxmin = {
            'W': None,  # optional weights
            'prox_A': prox_A,
            'prox_S': prox_S,
            'proxs_g': proxs_g,
            'steps_g': None,
            'Ls': None,
            'slack': 0.9,
            'update_order': None,
            'steps_g_update': 'steps_f',
            'max_iter': 100,
            'e_rel': 1e-3,
            'e_abs': 0,
            'traceback': None,
            'custom_prox_likelihood': modified_prox_likelihood
        }

        self.optimizer = BlockSDMM(self.Y, self.A0, self.S0,
                                   kwargs_proxmin=kwargs_proxmin)
