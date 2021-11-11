#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com

from oracle.solvers.Solver import Solver


class SolverCubicNewtonMiniBatch(Solver):

    def __init__(self, sub_solver, oracle, max_iter=1000, ro=0.1, l=0.5, epsilon=1e-3):
        super().__init__(oracle, ro, l, max_iter,epsilon )
        self.sub_solver = sub_solver

    def run(self, x_t, **kwargs):

        for i in range(self.max_iter):
            print("iteration: ", i)
            g_t, B_t = self.oracle.compute_oracle(x_t)
            delta, delta_m = self.sub_solver.solve(g_t, B_t, self.epsilon, self.ro, self.l)
            x_t1 = x_t + delta
            if delta_m >= -(self.epsilon ** 3 / self.ro) ** 0.5 / 100:
                delta = self.sub_solver.final_solve(g_t, B_t, self.epsilon, self.ro, self.l)
                return x_t + delta
            x_t = x_t1

        return x_t