#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
import numpy as np
from optimization.utils.Solver import Solver


class SolverStorm(Solver):

    def __init__(self, oracle, max_iter=1000, ):
        super().__init__(oracle, None, None, max_iter, None, None)

    def run(self, x_t, **kwargs):
        print("SolverStorm optimizing... ")
        objective_value_list = []
        for i in range(self.max_iter):
            self.oracle.update_sample()
            objective_value, g_t, _, eta = self.oracle.compute_oracle(x_t, )
            objective_value_list.append(objective_value)
            x_t = x_t - eta * g_t

        return x_t, np.array(objective_value_list)
