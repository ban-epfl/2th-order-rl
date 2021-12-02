#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
#
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# This implements the SGDHess for the gradient. as the same algorithm here https://arxiv.org/pdf/2103.03265.pdf
#
from abc import ABC
from typing import Any

import numpy as np

from optimization.oracles.LeastSquareOracle import LeastSquareOracle
from optimization.oracles.MeanOracle import MeanOracle
from optimization.utils.Solver import Solver


class SolverSGDHess1(Solver):

    def __init__(self, oracle: MeanOracle, max_iter, lr, n1, n2, G=100, momentum=0.9):
        self.G = G
        self.momentum = momentum
        super().__init__(oracle, None, None, max_iter, None, lr, n1, n2)

    def run(self,
            x_t: np.ndarray,
            **kwargs) -> (np.array, Any):
        print("SolverSGDHess optimizing... ")
        # initiate  momentum velocity
        v_t = np.zeros(x_t.shape)
        last_x_t = None
        for i in range(self.max_iter):
            self.oracle.update_sample(x_t)
            objective_value, g_t, B_t, _ = self.oracle.compute_oracle(x_t, )
            if i == 0:
                v_t = g_t
                last_x_t = x_t
                x_t = x_t - self.lr * g_t
                continue
            v_t = (1 - self.momentum) * (v_t + B_t(x_t - last_x_t)) + self.momentum * g_t
            norm_of_v_t = np.linalg.norm(v_t)
            if norm_of_v_t > self.G:
                v_t = self.G * v_t / norm_of_v_t

            self.oracle.log_changes(x_t, g_t)
            last_x_t=x_t
            x_t = x_t - self.lr * v_t

        return x_t
