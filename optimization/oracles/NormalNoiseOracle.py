#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com


import numpy as np

from optimization.utils.StochasticOracle import StochasticOracle


class NormalNoiseOracle(StochasticOracle):

    def compute_oracle(self, x_t, **kwargs):
        objective_value = self.objective_function.forward(x_t, self.s1).mean(axis=0)
        g_t = self.objective_function.gradient(x_t, self.s1).mean(axis=0)
        g_t += np.random.normal(0, 0.01, g_t.shape)
        hessian_noise = np.random.normal(0, 0.01, g_t.shape)

        def B_t(v):
            return (self.objective_function.hessian_vector(x_t, v, self.s2, self.r) + hessian_noise).mean(axis=0)

        return objective_value, g_t, B_t, None
