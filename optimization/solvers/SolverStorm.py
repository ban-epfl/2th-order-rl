#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com

from optimization.utils.Solver import Solver


class SolverStorm(Solver):

    def __init__(self, oracle, max_iter=1000, ro=0.1, l=0.5, epsilon=1e-3, lr=0.001):
        super().__init__(oracle, ro, l, max_iter, epsilon, lr)

    def run(self, x_t, **kwargs):
        eta=self.lr
        for i in range(self.max_iter):
            print("iteration: ", i)
            g_t, _, training_data = self.oracle.compute_oracle(x_t, lr=eta)
            eta = training_data["lr"]
            x_t = x_t - eta * g_t

        return x_t
