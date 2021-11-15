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

from optimization.utils.Solver import Solver


class SolverSGD(Solver):

    def __init__(self, oracle, max_iter=1000, ro=0.1, l=0.5, epsilon=1e-3, lr=0.001):
        super().__init__(oracle, ro, l, max_iter, epsilon, lr)

    def run(self, x_t, **kwargs):
        for i in range(self.max_iter):
            print("iteration: ", i)
            g_t, _, _ = self.oracle.compute_oracle(x_t, )
            x_t = x_t - self.lr * g_t
        return x_t
