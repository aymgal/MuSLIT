__author__ = 'aymgal'

import numpy as np
from proxmin import operators

from MuSLIT.transforms import starlet

def prox_id(X, step):
    return operators.prox_id(X, step)

def prox_plus(X, step):
    return operators.prox_plus(X, step)

def prox_column_norm(X, step):
    return operators.prox_unity(X, step, axis=1)

def prox_column_norm_plus(X, step):
    return prox_column_norm(prox_plus(X, step), step)

def prox_hard_thresh(X, step, thresh=0):
    return operators.prox_hard(X, step, thresh=thresh)

def prox_soft(X, step, thresh=0):
    return operators.prox_soft(X, step, thresh=thresh)
