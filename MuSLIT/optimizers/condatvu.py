__author__ = 'aymgal'


import numpy as np


class CondatVuOptimizer(object):


    def __init__(self, forward_op, forward_op_t, transform_op, transform_op_t,
                 prox1, prox2, n_iter, tau, nu):

        self.A  = forward_op
        self.At = forward_op_t

        self.B = lambda X, Y: -self.At( Y - self.A(X) )

        self.W  = transform_op
        self.Wt = transform_op_t

        self.prox1 = prox1
        self.prox2 = prox2

        self.tau = tau
        self.nu = nu


    def optimize(self, Y, X_0, U_0, n_iter=100):
        X = np.copy(X_0)
        U = np.copy(U_0)

        for i in range(n_iter):
            eval1 = X - self.tau * (self.W(U) + self.B(X, Y))
            X_new = self.prox1(eval1, self.tau)

            Y_new = 2 * X_new - X

            eval2 = U + self.eta * self.W(Y_new)
            U_new = self.prox2(eval2, self.eta)

            X = X_new
            U = U_new

        return X
