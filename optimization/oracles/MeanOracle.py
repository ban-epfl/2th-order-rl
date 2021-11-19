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

from optimization.utils.StochasticOracle import StochasticOracle


class MeanOracle(StochasticOracle):

    def compute_oracle(self, x_t, **kwargs):
        objective_value = self.objective_function.forward(x_t, self.s1).mean(axis=0)
        g_t = self.objective_function.gradient(x_t, self.s1).mean(axis=0)
        B_t = lambda v: 1 / self.n2 * self.objective_function.hessian_vector(x_t, v, self.s2, self.r).sum(axis=0)

        return objective_value, g_t, B_t, None
