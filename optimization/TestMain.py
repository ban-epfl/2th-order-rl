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

plot_names = []


def plot_objectiveValue_log(objective_values, plot_name):
    if not cumulative:
        plt.clf()
    else:
        plot_names.append(plot_name)

    plt.plot(range(len(objective_values)),
             objective_values)

    plt.xlabel('oracle calls ')
    plt.ylabel('objective value')
    plt.ylim([low_bound_y, high_bound_y])
    plt.xlim([low_bound_x, high_bound_x])
    plt.legend([plot_name], loc='upper right')
    if not cumulative:
        plt.savefig("plots/" + plot_name)


def plot_grad_log(grad_norm, plot_name):
    if not cumulative:
        plt.clf()
    else:
        plot_names.append(plot_name)

    plt.plot(range(len(grad_norm)),
             grad_norm)

    plt.ylim([0, high_bound_y])
    plt.xlabel('iteration')
    plt.ylabel('Norm')

    plt.legend(["grad_norm",], loc='upper right')
    if not cumulative:
        plt.savefig("plots/" + plot_name)


def plot_grad_log_tt(estimated_grads,grads, plot_name):
    if not cumulative:
        plt.clf()
    else:
        plot_names.append(plot_name)

    plt.plot(range(len(estimated_grads)),
             estimated_grads)
    plt.plot(range(len(grads)),
             grads)




    plt.xlabel('iteration')
    plt.ylabel('gradiant')

    plt.legend(["estimated_grad","true_grad",], loc='upper right')
    if not cumulative:
        plt.savefig("plots/" + plot_name)


def test_01():
    print("running test 1...")

    gd_solver = SubProblemCubicNewton(max_iter=15, c_prime=1)
    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.1, l=2, epsilon=1e-4, n1=500, n2=500)
    thetas = ms.run(np.random.RandomState(seed=42).rand(1))
    # plot the objective value list
    plot_objectiveValue_log(oracle.objective_values, "OneDimQuad_SolverCubicNewtonMiniBatch")
    print("best parameters", thetas, '\n')


def test_02():
    print("running test 2...")

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.01)
    oracle = MeanOracle(objective_function=ThreeDimQuad(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.001, l=4, epsilon=1e-5, n1=10000, n2=10000)
    objective_values = oracle.objective_values

    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(3)), '\n')


def test_03():
    print("running test 3...")
    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = SolverSGD(oracle=oracle, max_iter=100000, n1=1)
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    plot_objectiveValue_log(oracle.objective_values, "OneDimQuad_SolverSGD")
    print("best parameters", thetas, '\n')


def test_04():
    print("running test 4...")
    oracle = StormOracle(objective_function=OneDimQuad(), k=1e-1, c_factor=100, )
    ms = SolverStorm(oracle=oracle, max_iter=100000, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    plot_objectiveValue_log(oracle.objective_values, "OneDimQuad_SolverStorm")
    print("best parameters", thetas, '\n')


def test_05():
    print("running test 5...")

    gd_solver = SubProblemCubicNewton(max_iter=10, c_prime=0.1)
    oracle = NormalNoiseOracle(objective_function=WLooking(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=60000, ro=1, l=100, epsilon=1e-4, n1=100, n2=100)
    thetas = ms.run(np.random.RandomState(seed=45).rand(2))
    plot_objectiveValue_log(oracle.objective_values, "WLooking_SolverCubicNewtonMiniBatch")
    print("best parameters", thetas, '\n')


def test_06():
    print("running test 6...")

    oracle = NormalNoiseOracle(objective_function=WLooking(), )
    ms = SolverSGD(oracle=oracle, max_iter=60000, lr=1e-3, n1=100, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(2))
    objective_values = oracle.objective_values

    # plot the objective value list
    plot_objectiveValue_log(oracle.objective_values, "WLooking_SolverSGD")
    plt.savefig("plots/WLooking_SolverSGD")
    print("best parameters", thetas, '\n')


def test_07():
    print("running test 7...")

    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = GradientLeastSquares(oracle=oracle, j1=1, j2=1, l=1, alpha=0.99, max_iter=4000, lr=1e-3 )
    thetas, estimated_grads, true_grads = ms.run(np.random.RandomState(seed=45).rand(1))

    # plot the objective value list
    plot_objectiveValue_log(oracle.objective_values, "OneDimQuad_GradientLeastSquares")
    plot_grad_log(oracle.estimated_norm_gradients, "normOneDimQuad_GradientLeastSquares" )

    # plot_grad_log_tt(estimated_grads, true_grads, "gradOneDimQuad_GradientLeastSquares" )
    print("best parameters", thetas, '\n')


cumulative = False
high_bound_x = 5000
high_bound_y = 3
low_bound_x = -1
low_bound_y = 0

# test_01()
# test_03()
# test_04()
# test_05()
# test_06()
test_07()

if cumulative:
    plt.legend(plot_names, loc='upper right')
    plt.savefig("plots/" + '-'.join(plot_names))
