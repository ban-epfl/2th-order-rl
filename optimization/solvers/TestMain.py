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

from optimization.oracles.StormOracle import StormOracle
from optimization.solvers.SolverSGD import SolverSGD
from optimization.solvers.SolverStorm import SolverStorm
from optimization.utils.Module import Module
from optimization.solvers.SolverCubicNewtonMiniBatch import SolverCubicNewtonMiniBatch
from optimization.oracles.StochasticOracle import StochasticOracle
from optimization.solvers.SubProblemCubicNewton import SubProblemCubicNewton


def test_01():
    print("running test 1...")

    class MyModule(Module):

        def get_sample_dim(self) -> int:
            return 3

        def forward(self, x, z):
            return z[:, 0] * x ** 2 + z[:, 1] * x + z[:, 2]

        def gradient(self, x, z):
            return np.expand_dims(2 * z[:, 0] * x + z[:, 1], axis=1)

        def hessian_vector(self, x_t, v, z, r):
            return np.matmul(np.expand_dims(2 * z[:, 0],axis=1), np.expand_dims(v,axis=1))

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.01)
    oracle = StochasticOracle(module=MyModule(), n1=10000, n2=10000)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.001, l=4, epsilon=1e-5)
    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(1)))



def test_02():
    print("running test 2...")

    class MyModule(Module):

        def get_sample_dim(self) -> int:
            return 3

        def forward(self, x, z):
            return np.matmul(z[:, 0].reshape(-1,1) , (x**2).reshape(1,-1)).sum(axis=1) +\
                                                                         np.matmul(z[:, 1].reshape(-1,1), x.reshape(1,-1)).sum(axis=1) +\
                                                                         z[:, 2]

        def gradient(self, x, z):
            return np.expand_dims(np.matmul(2*z[:, 0].reshape(-1,1), x.reshape(1,-1)).sum(axis=1) + z[:, 1], axis=1)

        def hessian_vector(self, x_t, v, z, r):
            return np.matmul(np.expand_dims(2 * z[:, 0],axis=1), np.expand_dims(v,axis=1))

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.01)
    oracle = StochasticOracle(module=MyModule(), n1=10000, n2=10000)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.001, l=4, epsilon=1e-5)
    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(3)))





def test_03():
    print("running test 3...")

    class MyModule(Module):

        def get_sample_dim(self) -> int:
            return 3

        def forward(self, x, z):
            return z[:, 0] * x ** 2 + z[:, 1] * x + z[:, 2]

        def gradient(self, x, z):
            return np.expand_dims(2 * z[:, 0] * x + z[:, 1], axis=1)

        def hessian_vector(self, x_t, v, z, r):
            return np.matmul(np.expand_dims(2 * z[:, 0],axis=1), np.expand_dims(v,axis=1))

    oracle = StochasticOracle(module=MyModule(), n1=1, n2=1)
    ms = SolverSGD( oracle=oracle, max_iter=2000, ro=0.001, l=4, epsilon=1e-5)
    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(1)))

test_03()




def test_04():
    print("running test 4...")

    class MyModule(Module):

        def get_sample_dim(self) -> int:
            return 3

        def forward(self, x, z):
            return z[:, 0] * x ** 2 + z[:, 1] * x + z[:, 2]

        def gradient(self, x, z):
            return np.expand_dims(2 * z[:, 0] * x + z[:, 1], axis=1)

        def hessian_vector(self, x_t, v, z, r):
            return np.matmul(np.expand_dims(2 * z[:, 0],axis=1), np.expand_dims(v,axis=1))

    oracle = StormOracle(module=MyModule(), n1=1, n2=1)
    ms = SolverStorm( oracle=oracle, max_iter=3000, ro=0.001, l=4, epsilon=1e-5, lr=1e-5)
    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(1)))

test_04()
