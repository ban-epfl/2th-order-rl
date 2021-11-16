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
        g_t = self.module.gradient(x_t, s1).mean(axis=0)
        g_t += np.random.normal(0, 1, g_t.shape)
        B_t = lambda v: 1 / self.n2 * (
                    self.module.hessian_vector(x_t, v, s2, self.r) + np.random.normal(0, 1, g_t.shape)).sum(axis=0)

        return g_t, B_t, None
