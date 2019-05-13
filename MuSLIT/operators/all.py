__author__ = 'aymgal'

import numpy as np
import proxmin

from MuSLIT.transforms import starlet


def prox_plus(X, step):
    return proxmin.operators.prox_plus(X, step)

def prox_column_norm(X, step):
    return proxmin.operators.prox_unity(X, step, axis=1)

def prox_column_norm_plus(X, step):
    return prox_column_norm(prox_plus(X, step), step)

def prox_hard_thresh(X, step, thresh=0):
    reeturn proxmin.operators.prox_hard(X, step, thresh=thresh)