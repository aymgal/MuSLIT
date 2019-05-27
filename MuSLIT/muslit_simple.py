__author__ = 'aymgal'

import numpy as np
from scipy import signal as scp
import functools

import MuSLIT.utils.image as image_utils
import MuSLIT.utils.math as math_utils
import MuSLIT.operators.proximals as proxs
from MuSLIT.optimizers.condatvu import CondatVuOptimizer
from MuSLIT.optimizers.threshold_scheduler import ThresholdScheduler
from MuSLIT.transforms.data_structures import ComponentMatrix



class LightModellerSimple(object):

    """TODO : check matrix shapes all over the place"""

    def __init__(self, target_image, forward_model, dictionnary_set, 
                 threshold_mode='linear', optimizer_mode='condatvu',
                 source_to_image_ratio=1):
        # number of components to deblend
        self.num_components = mixing_matrix.num_components

        # number of spectral bands
        self.num_bands = target_image.num_bands

        # number of side pixels in lens and source plan e
        self.num_pix = target_image.num_pix
        self.num_pix_src = int(self.num_pix * source_to_image_ratio)

        self._target_image    = target_image
        self._forward_model   = forward_model
        self._dictionnary_set = dictionnary_set

        self._init_threshold(threshold_mode)
        self._init_optimizer(optimizer_mode)


    def run(self, n_iter=100):
        self._random_init()
        self.optimizer(self.X0, self.U0, n_iter=n_iter)


    def _prox_soft_thresh(self, component_matrix, step, tresh=0):
        cm = component_matrix
        cm.lens_data = proxs.prox_soft_thresh(cm.lens_data, s, thresh=thresh)
        cm.source_data = proxs.prox_soft_thresh(cm.source_data, s, thresh=thresh)
        return cm

    def _prox_soft_thresh_dual(self, component_matrix, step, tresh=0):
        cm = component_matrix
        cm_copy = copy.deepcopy(cm)
        cm_primal = self.prox_soft_thresh(cm_copy, step, thresh=thresh)
        cm = cm - cm_primal
        return cm

    def _prox_plus(self, component_matrix, step, tresh=0):
        cm = component_matrix
        cm.lens_data = proxs.prox_plus(cm.lens_data, s)
        cm.source_data = proxs.prox_plus(cm.source_data, s)
        return cm

    def _init_threshold(self, threshold_mode):
        # TODO estimate from noise levels in 'target_image'
        init_value = 100
        mb_noise = self._target_image.noise_MAD
        final_value = 3 * np.min(mb_noise)  # TODO : one level per band and per component
        self._thresh_scheduler = ThresholdScheduler(init_value=init_value,
                                                    final_value=final_value,
                                                    mode=threshold_mode)

    def _init_optimizer(self, optimizer_mode):
        #TODO different optimizers ? (optimizer_mode)

        forward_op   = self._forward_model.operator
        forward_op_t = self._forward_model.transpose
        transform_op   = self._dictionnary_set.analysis
        transform_op_t = self._dictionnary_set.synthesis

        prox_g1 = self._prox_plus
        prox_g2_dual = self._prox_soft_thresh_dual

        Lip = self._forward_model.lipschitz
        tau = 1. / Lip
        eta = 0.5 * Lip
        print("INFO :\nLip={}\ntau={}\neta={}".format(Lip, tau, eta))

        self.optimizer = CondatVuOptimizer(self._target_image.data, 
                                           forward_op, forward_op_t, 
                                           transform_op, transform_op_t,
                                           self._thresh_scheduler,
                                           prox_g1, prox_g2_dual, 
                                           tau, nu)

    def _random_init(self):
        # TODO use num_components for flexibility

        U0_l = LightComponent(self.num_pix, random_init=True)
        U0_s = LightComponent(self.num_pix_src, random_init=True)
        self.U0 = ComponentMatrix([U0_l.data, U0_s.data])
        
        X0_l = LightComponent(self.num_pix, random_init=True)
        X0_s = LightComponent(self.num_pix_src, random_init=True)
        self.X0 = ComponentMatrix([X0_l.data, X0_s.data])
