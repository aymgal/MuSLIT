__author__ = 'aymgal'


import numpy as np


class ThresholdScheduler(object):


    def __init__(self, init_value=1, final_value=0, mode='linear'):
        self._mode = mode
        self._init_val  = init_value
        self._final_val = final_value


    def at_iter(self, i, **kwargs):
        if self._mode == 'constant':
            return self._constant(i, **kwargs)
        elif self._mode == 'linear':
            return self._linear_decrease(i, **kwargs)
        else:
            raise NotImplementedError


    def _constant(self, i, init_value=None):
        if init_value is None:
            val = self._init_val
        return val


    def _linear_decrease(self, i, init_value=None, n_iter=100, c=5):
        if c >= n_iter:
         if n_iter
        if init_value is None:
            init_value = self._init_val
        val = (init_value - self._final_val) / (n_iter - c - i)
        if val < 0 or val <= self._final_val:
            val = self._final_val
        return val
