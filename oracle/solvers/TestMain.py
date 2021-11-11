#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# Here we put the general tests.

import numpy as np
from oracle.solvers.Module import Module
from oracle.solvers.SolverCubicNewtonMiniBatch import SolverCubicNewtonMiniBatch
from oracle.solvers.StochasticOracle import StochasticOracle
from oracle.solvers.SubProblemCubicNewton import SubProblemCubicNewton


def test_01():
    print("running test 1...")

    class MyModule(Module):

        def get_param_num(self) -> int:
            return 3

        def forward(self, x, z):
            return z[:, 0] * x ** 2 + z[:, 1] * x + z[:, 2]

        def gradient(self, x, z):
            return np.expand_dims(2 * z[:, 0] * x + z[:, 1], axis=1)

        def hessian_vector(self, x_t, v, z, r):
            return np.matmul(np.expand_dims(2 * z[:, 0],axis=1), np.expand_dims(v,axis=1))

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.001)
    oracle = StochasticOracle(module=MyModule(), n1=10000, n2=10000)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.00001, l=1)
    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(1)))


test_01()
