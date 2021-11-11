#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# "StochasticOracle" is a base class that requires implementation of functions
# to return the value, gradient, symmetric sparse hessian and hessian vector product.
# The functions to get value/derivate information here are "virtual" and need to be
# implemented for specific functions.

import numpy as np

from oracle.solvers.utils.Oracle import Oracle


class StochasticOracle(Oracle):

    def compute_oracle(self, x_t):

        s1 = np.random.normal(1, 2, (self.n1, self.module.get_param_num()))
        g_t = self.module.gradient(x_t, s1).reshape((self.n1, 1)).mean(axis=0)
        s2 = np.random.normal(1, 2, (self.n2, self.module.get_param_num()))
        B_t = lambda v: 1 / self.n2 * self.module.hessian_vector(x_t, v, s2, self.r).sum(axis=0)

        return g_t, B_t
