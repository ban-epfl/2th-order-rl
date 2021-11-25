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

from optimization.oracles.LeastSquareOracle import LeastSquareOracle
from optimization.utils.Solver import Solver


# from sklearn import datasets, linear_model


class GradientLeastSquares(Solver):

    def __init__(self, oracle: LeastSquareOracle, lr=0.001, l=1, j1=100, j2=100, alpha=0.1, max_iter=10000,
                 point_limit=300):
        self.j1 = j1
        self.j2 = j2
        self.alpha = alpha
        self.x_js = []
        self.point_limit = point_limit
        super().__init__(oracle, None, l, max_iter, None, lr, 1, 0)

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = np.matmul(X, theta)
        return 1 / (2 * m) * np.linalg.norm(h - y, ord=2) ** 2

    def least_squares_SGD(self, c, A, theta, num_iters=100, gamma=0.001, ):
        m = c.shape[0]
        models = []
        for iter in range(num_iters):
            random_idx = np.random.randint(m)
            # update weights
            theta = theta - gamma * np.transpose(A) * (np.matmul(A, theta) - c)
            models.append(np.copy(theta))
            # best_model_idx = np.argmin(val_loss)
        return models[-1]

    def get_x_js(self, x_t):
        """
        Args:
            x_t: np.ndarray

        Returns:
         a matrix with the shape of (self.j1, x_t.shape[0]) OR (self.j1, param_num)

        """
        return x_t + np.random.normal(0, 0.01, (self.j1, x_t.shape[0]))

    def run(self, x_t, **kwargs):
        print("GradientLeastSquares optimizing... ")
        self.x_js.append(x_t)
        self.oracle.update_sample(x_t)
        for i in range(self.max_iter):

            # x_js = self.get_x_js(x_t)
            x_js = np.array(self.x_js)
            if i > 0: self.x_js.append(x_t)

            x_js_minus = x_js - x_t
            x_js_minus = np.c_[np.ones(x_js.shape[0]), x_js_minus]
            A = self.alpha / x_js.shape[0] * np.matmul(x_js_minus.T, x_js_minus) + (1 - self.alpha) * np.identity(
                x_t.shape[0] + 1)


            # compute lipschitz part of the right hand side of the equation Ad=c
            lips_sum = 0
            min_range = 0
            if x_js.shape[0] > self.point_limit:
                min_range = x_js.shape[0] - self.point_limit
            for j in range(min_range, x_js.shape[0]):
                objective_value_x_j, _, _, _ = self.oracle.compute_index_oracle(x_js[j], j)
                objective_value_x_t, _, _, _ = self.oracle.compute_index_oracle(x_t, j)
                lips_sum += (self.alpha / x_js.shape[0]) * (
                        objective_value_x_j - self.l / 2 * np.linalg.norm(x_js[j] - x_t,  ord=2) ** 2) * x_js_minus[j]
            b = lips_sum

            # compute mean of gradient estimate of the right hand side of the equation Ad=c
            gradient_sum = np.zeros(x_t.shape)
            for j in range(self.j2):
                self.oracle.update_sample(x_t)
                objective_value_x_t, g_t, _, _ = self.oracle.compute_oracle(x_t, )
                gradient_sum += g_t

            gradient_sum = np.concatenate([[0],gradient_sum])
            b += (1 - self.alpha) / self.j2 * gradient_sum


            # x_js_minus= x_js - x_t x_js_minus_transpose=x_js_minus.T for s in range(x_t.shape[0]): row= np.zeros(
            # x_t.shape[0]) for k in range(x_t.shape[0]): if s==k: row[k]= self.alpha / x_js.shape[0] * np.dot(
            # x_js_minus_transpose[s], x_js_minus[k])+1-self.alpha else: row[k] = self.alpha / x_js.shape[0] *
            # np.dot(x_js_minus_transpose[s], x_js_minus[k])
            #
            #     self.least_squares_SGD(b[s], row, None)

            delta = np.linalg.lstsq(A, b, rcond=None)[0]
            delta=delta[1:]
            self.oracle.log_gradient(x_t, delta)

            x_t = x_t - self.lr * delta
        return x_t
