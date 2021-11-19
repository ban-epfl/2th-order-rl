#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# This implements the least squares estimator for the gradient. Possible field of
# this class are the past iterates of the master problem, the convex combination 
# parameter, the number of samples and other precision parameters.
#


import numpy as np
from optimization.utils.Solver import Solver
from sklearn import datasets, linear_model


class GradientLeastSquares(Solver):

    def __init__(self, oracle,  lr=0.001, l=1, j1=100, j2=100, alpha= 0.1):
        self.j1 = j1
        self.j2 = j2
        self.alpha = alpha
        super().__init__(oracle, None, l, None, None, lr)

    def get_x_js(self, x_t):
        """
        Args:
            x_t: np.ndarray

        Returns:
         a matrix with the shape of (self.j1, x_t.shape[0]) OR (self.j1, param_num)

        """
        return x_t + np.random.rand(self.j1, x_t.shape[0])

    def run(self, x_t, **kwargs):
        print("GradientLeastSquares optimizing... ")
        objective_value_list = []
        x_js = self.get_x_js(x_t)
        A = self.alpha / self.j1 * np.matmul((x_js - x_t).T, (x_js - x_t)) + (1 - self.alpha) * np.identity(
            x_t.shape[0])

        # compute lipschitz part of the right hand side of the equation Ad=c
        lips_sum = 0
        for j in range(self.j1):
            self.oracle.update_sample()
            objective_value_x_j, _, _, _ = self.oracle.compute_oracle(x_js[j], )
            objective_value_x_t, _, _, _ = self.oracle.compute_oracle(x_t, )
            lips_sum += objective_value_x_j - objective_value_x_t - self.l * np.linalg.norm(x_js[j] - x_t, ord=2) ** 2
        b = self.alpha / self.j1 * lips_sum

        # compute mean of gradient estimate of the right hand side of the equation Ad=c
        gradient_sum = np.zeros(x_t.shape)
        for j in range(self.j2):
            self.oracle.update_sample()
            _, g_t, _, _ = self.oracle.compute_oracle(x_t, )
            gradient_sum += g_t
        b += (1 - self.alpha) / self.j2 * gradient_sum
        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(A, b)
        regr.
        return np.linalg.lstsq(A, b,)[0], None
