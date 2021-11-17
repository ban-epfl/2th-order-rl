#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# Here we put the general tests.
import sys

import numpy as np
import matplotlib.pyplot as plt
from optimization.oracles.NormalNoiseOracle import NormalNoiseOracle
from optimization.oracles.StochasticOracle import StochasticOracle
from optimization.oracles.StormOracle import StormOracle
from optimization.test_modules import WLooking, OneDimQuad, ThreeDimQuad
from optimization.solvers.SolverSGD import SolverSGD
from optimization.solvers.SolverStorm import SolverStorm
from optimization.solvers.SolverCubicNewtonMiniBatch import SolverCubicNewtonMiniBatch
from optimization.solvers.SubProblemCubicNewton import SubProblemCubicNewton
from optimization.utils.Oracle import Oracle


def test_01():
    print("running test 1...")

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.0001)
    oracle = StochasticOracle(module=OneDimQuad(), n1=1000, n2=1000)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.0001, l=4, epsilon=1e-5)
    thetas, objective_values= ms.run(np.random.RandomState(seed=42).rand(1))

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
    oracle = StochasticOracle(module=ThreeDimQuad(), n1=10000, n2=10000)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.001, l=4, epsilon=1e-5)
    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(3)),'\n')


def test_03():
    print("running test 3...")
    oracle = StochasticOracle(module=OneDimQuad(), n1=1000, n2=1)
    ms = SolverSGD(oracle=oracle, max_iter=9000,)
    thetas, objective_values = ms.run(np.random.RandomState(seed=45).rand(1))
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
    oracle = StormOracle(module=OneDimQuad(), k=1e-2, c_factor=100, n1=1 )
    ms = SolverStorm(oracle=oracle, max_iter=9000,)
    thetas, objective_values = ms.run(np.random.RandomState(seed=45).rand(1))
    # plot the objective value list
    plt.clf()
    plt.plot(1e-6 * np.array(range(len(objective_values[5:]))),
             objective_values[5:])
    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    # plt.legend(['Storm', ], loc='upper right')
    # plt.savefig("plots/OneDimQuad_SolverStorm")
    print("best parameters", thetas, '\n')


def test_05():
    print("running test 5...")

    gd_solver = SubProblemCubicNewton(max_iter=10, c_prime=0.1)
    oracle = NormalNoiseOracle(module=WLooking(), n1=100, n2=100)
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=60000, ro=1, l=100, epsilon=1e-4, )
    thetas, objective_values = ms.run(np.random.RandomState(seed=45).rand(2))
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

    oracle = NormalNoiseOracle(module=WLooking(), n1=100, n2=100)
    ms = SolverSGD( oracle=oracle, max_iter=60000, lr=1e-3)
    thetas, objective_values= ms.run(np.random.RandomState(seed=45).rand(2))
    # plot the objective value list
    plt.clf()
    plt.plot(1e-4*np.array(range(len(objective_values)))[150:600],
             objective_values[150:600])

    plt.xlabel('oracle calls (1e6)')
    plt.ylabel('objective value')
    plt.legend(['SGD'], loc='upper right')
    plt.savefig("plots/WLooking_SolverSGD")
    print("best parameters",thetas ,'\n')


test_05()
test_06()
