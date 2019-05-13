__author__ = 'aymgal'

import MuSLIT.operators.all as all_ops
from MuSLIT.optimizers.bsdmm import BlockSDMM
from MuSLIT.tranforms.starlet import StarletTransform
import MuSLIT.utils.image as image_utils


class LensLightModeller(object):

    """TODO : check matrix shapes all over the place"""


    def __init__(self, target_image, lensing_matrix, mode='synthesis', 
                 thresh=0, starlet_level=1, num_components=2, 
                 source_to_image_ratio=1):

        num_pix1, num_pix2 = target_image.shape
        assert num_pix1 == num_pix2, "Inout image must be square"

        self.num_pix = num_pix1
        self.num_pix_src = int(self.num_pix * source_to_image_ratio)
        self.n2  = self.num_pix**2
        self.ns2 = self.num_pix_src**2

        self.num_bands = target_image.shape[-1]
        self.num_components = num_components

        self.lensing_matrix = lensing_matrix

        dirac = np.ones((self.num_pix, self.num_pix))
        self.lensed = self.image_to_source(dirac, lensed=None)

        self.F     = lambda X: self.source_to_image(X)
        self.F_inv = lambda X: self.image_to_source(X, lensed=lensed)

        self.Y  = image_utils.multiband_to_array(target_image)
        self.A0 = np.rand.rand((self.num_bands, self.num_components))
        self.S0 = np.rand.rand((self.num_components, self.n2))

        starlet = StarletTransform(lvl=starlet_level)

        prox_A = all_ops.prox_plus
        prox_S = all_ops.prox_plus

        proxs_g = [
            # ops for A
            [
                all_ops.prox_column_norm,
            ],
            # ops for S
            [
                self.prox_img_plane_starlet_soft_thresh,
                self.prox_src_plane_starlet_soft_thresh,
            ]
        ]

        kwargs_optim = {
            'W': None,  # optional weights
            'prox_A': prox_A,
            'prox_S': prox_S,
            'proxs_g': proxs_g,
            'steps_g': None,
            'Ls': None,
            'slack': 0.9,
            'update_order': None,
            'steps_g_update': 'steps_f',
            'max_iter': 300,
            'e_rel': 1e-3,
            'e_abs': 0,
            'traceback': None
        }

        self.optimizer = BlockSDMM(kwargs_optim)


    def run(self):
        A, S, hist = self.optimizer.optimize()
        return A, S, hist


    def prox_src_plane_starlet_soft_thresh(self, X, step):
        """X is altered by this function !"""
        FS = X[0, :]
        S = self.F_inv(FS)
        S_coeffs = starlet.tranform(S)
        S_coeffs_t = all_ops.prox_soft_thresh(S_coeffs, step, thresh=thresh)
        S_t = starlet.inverse(coeffs_t)
        X[0, :] = S_t
        return X


    def prox_img_plane_starlet_soft_thresh(self, X, step):
        """X is altered by this function !"""
        G = X[1, :]
        G_coeffs = starlet.tranform(G)
        G_coeffs_t = all_ops.prox_soft_thresh(G_coeffs, step, thresh=thresh)
        G_t = starlet.inverse(coeffs_t)
        X[1, :] = G_t
        return X


    def source_to_image(self, source, ones=True):
        if ones:
            one_lens = self.source_to_image(np.ones(source.shape), 
                                            self.lensing_matrix, ones=False)
            one_lens[np.where(one_lens == 0)] = 1
        else:
            one_lens = 1.

        image = np.zeros((self.num_pix, self.num_pix))
        xb, yb = image_utils.square_grid(self.num_pix_src)

        k = 0
        for pos in self.lensing_matrix:
            if np.size(np.shape(pos)) != 1:
                image[np.array(pos[0][:]),
                      np.array(pos[1][:])] += source[xb[k], yb[k]]
            k += 1
        return image / one_lens


    def image_to_source(self, image, square=False, lensed=None):
        source = np.zeros((self.num_pix_src, self.num_pix_src))

        xb, yb = image_utils.square_grid(self.num_pix_src)

        for k in range(self.num_pix_src**2):
            pos = self.lensing_matrix[k]
            if np.size(np.shape(pos)) > 1:

                light = Image[np.array(pos[0][:]), np.array(pos[1][:])]

                if lensed is not None:
                    light /= np.max([1, np.size(pos[0][:])])
                    if square:
                        source[xb[k], yb[k]] += np.sum(light**2)
                    else:
                        source[xb[k], yb[k]] += np.sum(light)
                else:
                    source[xb[k],yb[k]] += np.sum(light)
        if square:
            source = np.sqrt(source)
        return source
