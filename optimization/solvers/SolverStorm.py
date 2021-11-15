#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com

from optimization.utils.Solver import Solver


class SolverStorm(Solver):

    def __init__(self, oracle, max_iter=1000, ro=0.1, l=0.5, epsilon=1e-3, learning_rate=0.001):
        super().__init__(oracle, ro, l, max_iter, epsilon, learning_rate)

    def run(self, x_t, **kwargs):
        for i in range(self.max_iter):
            print("iteration: ", i)
            g_t, _, training_data = self.oracle.compute_oracle(x_t, lr=self.learning_rate)
            self.learning_rate = training_data["learning_rate"]
            x_t = x_t - training_data["learning_rate"] * g_t

        return x_t
