#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com


import numpy as np

from optimization.utils.Oracle import Oracle


class NormalNoiseOracle(Oracle):

    def compute_oracle(self, x_t, **kwargs):
        s1, s2 = self.module.get_samples(self.n1, self.n2)
        objective_value = self.module.forward(x_t, s1).mean(axis=0)
        g_t = self.module.gradient(x_t, s1).mean(axis=0)
        g_t += np.random.normal(0, 0.01, g_t.shape)
        hessian_noise = np.random.normal(0, 0.01, g_t.shape)

        def B_t(v):
            return (self.module.hessian_vector(x_t, v, s2, self.r) + hessian_noise).mean(axis=0)

        return objective_value, g_t, B_t, None
