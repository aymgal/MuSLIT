__author__ = 'aymgal'


import numpy as np


class CondatVuOptimizer(object):


    def __init__(self, target_data, forward_op, forward_op_t, 
                 transform_op, transform_op_t, thresh_schedule,
                 prox1, prox2, n_iter, tau, nu):

        self.Y = target_data

        self.A  = forward_op    # A(·) = HA·F
        self.At = forward_op_t  # (HAF)^T(·) = F^T · A^T H^T

        self.B = lambda X: -self.At( self.Y - self.A(X) )   # grad of ||Y-Ax||^2

        self.W  = transform_op      # Phi^T
        self.Wt = transform_op_t    # Phi

        self.prox1 = prox1
        self.prox2 = prox2

        self.tau = tau
        self.nu = nu

        self.thresh_schedule = thresh_schedule

    def optimize(self, X_0, U_0, n_iter=100):
        """X_0, U_0 must be ComponentMatrix objects"""
        X = X_0
        U = U_0
        for i in range(n_iter):
            thresh = self.thresh_schedule(i)

            eval1 = X - self.tau * (self.W(U) + self.B(X))
            X_new = self.prox1(eval1, self.tau * thresh)

            Y_new = 2 * X_new - X

            eval2 = U + self.eta * self.W(Y_new)
            U_new = self.prox2(eval2, self.eta * thresh)

            X = X_new
            U = U_new
        return X
