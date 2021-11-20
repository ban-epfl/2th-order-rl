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
import matplotlib.pyplot as plt
from optimization.oracles.NormalNoiseOracle import NormalNoiseOracle
from optimization.oracles.MeanOracle import MeanOracle
from optimization.oracles.StormOracle import StormOracle
from optimization.solvers.GradientLeastSquares import GradientLeastSquares
from optimization.test_modules import WLooking, OneDimQuad, ThreeDimQuad
from optimization.solvers.SolverSGD import SolverSGD
from optimization.solvers.SolverStorm import SolverStorm
from optimization.solvers.SolverCubicNewtonMiniBatch import SolverCubicNewtonMiniBatch
from optimization.solvers.SubProblemCubicNewton import SubProblemCubicNewton


def test_01():
    print("running test 1...")

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.0001)
    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.0001, l=4, epsilon=1e-5, n1=1000, n2=1000)
    thetas= ms.run(np.random.RandomState(seed=42).rand(1))
    objective_values=oracle.objective_values
    # plot the objective value list
    plt.plot(range(len(objective_values)),
             objective_values)
    plt.xlabel('iteration')
    plt.ylabel('objective value')
    # plt.show()
    plt.savefig("plots/OneDimQuad_SolverCubicNewtonMiniBatch")
    print("best parameters",thetas ,'\n')

def test_02():
    print("running test 2...")

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.01)
    oracle = MeanOracle(objective_function=ThreeDimQuad(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.001, l=4, epsilon=1e-5, n1=10000, n2=10000)
    objective_values=oracle.objective_values

    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(3)),'\n')


def test_03():
    print("running test 3...")
    oracle = MeanOracle(objective_function=OneDimQuad(),)
    ms = SolverSGD(oracle=oracle, max_iter=9000,  n1=1)
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    objective_values=oracle.objective_values

    # plot the objective value list
    plt.clf()
    plt.plot(1e-6 * np.array(range(len(objective_values[5:]))),
             objective_values[5:])
    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    plt.legend(['Storm', ], loc='upper right')
    plt.savefig("plots/OneDimQuad_SolverSGD")
    print("best parameters", thetas, '\n')


def test_04():
    print("running test 4...")
    oracle = StormOracle(objective_function=OneDimQuad(), k=1e-2, c_factor=100,)
    ms = SolverStorm(oracle=oracle, max_iter=9000, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    objective_values=oracle.objective_values

    # plot the objective value list
    plt.clf()
    plt.plot(1e-6 * np.array(range(len(objective_values[5:]))),
             objective_values[5:])
    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    # plt.legend(['Storm', ], loc='upper right')
    plt.savefig("plots/OneDimQuad_SolverStorm")
    print("best parameters", thetas, '\n')


def test_05():
    print("running test 5...")

    gd_solver = SubProblemCubicNewton(max_iter=10, c_prime=0.1)
    oracle = NormalNoiseOracle(objective_function=WLooking(),)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=60000, ro=1, l=100, epsilon=1e-4,  n1=100, n2=100)
    thetas = ms.run(np.random.RandomState(seed=45).rand(2))
    objective_values=oracle.objective_values

    # plot the objective value list
    plt.clf()
    plt.plot(1e-4*np.array(range(len(objective_values)))[150:600],
             objective_values[150:600])
    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    plt.legend(['Cubic',  ], loc='upper right')
    plt.savefig("plots/WLooking_SolverCubicNewtonMiniBatch")
    print("best parameters", thetas, '\n')

def test_06():
    print("running test 6...")

    oracle = NormalNoiseOracle(objective_function=WLooking(), n1=100, n2=100)
    ms = SolverSGD( oracle=oracle, max_iter=60000, lr=1e-3)
    thetas= ms.run(np.random.RandomState(seed=45).rand(2))
    objective_values=oracle.objective_values

    # plot the objective value list
    plt.clf()
    plt.plot(1e-4*np.array(range(len(objective_values)))[150:600],
             objective_values[150:600])

    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    plt.legend(['SGD'], loc='upper right')
    plt.savefig("plots/WLooking_SolverSGD")
    print("best parameters",thetas ,'\n')


def test_07():
    print("running test 7...")

    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = GradientLeastSquares( oracle=oracle, j1=50, j2=50, l=0.1, alpha=0.70, max_iter=1000,)
    thetas= ms.run(np.random.RandomState(seed=45).rand(1))
    objective_values=oracle.objective_values

    # plot the objective value list
    plt.clf()
    plt.plot(1e-4*np.array(range(len(objective_values)))[150:600],
             objective_values[150:600])

    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    plt.legend(['GradientLeastSquares'], loc='upper right')
    plt.savefig("plots/OneDimQuad_GradientLeastSquares")
    print("best parameters",thetas ,'\n')


# test_05()
test_03()
test_04()
