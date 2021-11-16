#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# This class implements the SGD method. Fields of this class are
# stochastic orable, max iterations, learning rate.
#
import inspect
import sys

import numpy as np

from optimization.utils.Solver import Solver
import matplotlib.pyplot as plt


class SolverSGD(Solver):

    def __init__(self, oracle, max_iter=1000, ro=0.1, l=0.5, epsilon=1e-3, lr=0.001):
        super().__init__(oracle, ro, l, max_iter, epsilon, lr)

    def run(self, x_t, **kwargs):
        print("SolverSGD optimizing... ")
        objective_value_list=[]
        for i in range(self.max_iter):
            objective_value, g_t, _, _ = self.oracle.compute_oracle(x_t, )
            objective_value_list.append(objective_value)
            x_t = x_t - self.lr * g_t
        return x_t, np.array(objective_value_list)
