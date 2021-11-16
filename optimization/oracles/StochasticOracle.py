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

from optimization.utils.Oracle import Oracle


class StochasticOracle(Oracle):

    def compute_oracle(self, x_t, **kwargs):
        s1, s2 = self.module.get_samples(self.n1, self.n2)
        objective_value = self.module.forward(x_t, s1).mean(axis=0)
        g_t = self.module.gradient(x_t, s1).mean(axis=0)
        B_t = lambda v: 1 / self.n2 * self.module.hessian_vector(x_t, v, s2, self.r).sum(axis=0)

        return objective_value, g_t, B_t, None
