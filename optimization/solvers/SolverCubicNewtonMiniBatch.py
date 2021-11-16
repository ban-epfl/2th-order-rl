#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
import numpy as np

from optimization.utils.Solver import Solver


class SolverCubicNewtonMiniBatch(Solver):

    def __init__(self, sub_solver, oracle, max_iter=1000, ro=0.1, l=0.5, epsilon=1e-3, lr=0.001):
        super().__init__(oracle, ro, l, max_iter, epsilon, lr)
        self.sub_solver = sub_solver

    def run(self, x_t, **kwargs):
        print("SolverCubicNewtonMiniBatch optimizing... ")
        objective_value_list=[]
        for i in range(self.max_iter):
            print("iteration= ",i)
            objective_value, g_t, B_t, _ = self.oracle.compute_oracle(x_t, )
            objective_value_list.append(objective_value)
            print("iteration1= ",i)
            delta, delta_m = self.sub_solver.solve(g_t, B_t, self.epsilon, self.ro, self.l)
            print("iteration2= ",i)

            x_t1 = x_t + delta
            if delta_m >= -(self.epsilon ** 3 / self.ro) ** 0.5 / 100:
                delta = self.sub_solver.final_solve(g_t, B_t, self.epsilon, self.ro, self.l)
                return x_t + delta, np.array(objective_value_list)
            x_t = x_t1

        return x_t, np.array(objective_value_list)
