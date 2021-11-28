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

from optimization.oracles.LeastSquareOracle import LeastSquareOracle
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


def plot_objective_value(objective_values, plot_name):
    if not cumulative:
        axis[0].clf()
    else:
        plot_names.append(plot_name)

    axis[0].plot(range(len(objective_values)),
             objective_values)

    axis[0].set_xlabel('oracle calls ')
    axis[0].set_ylabel('obj value')
    axis[0].set_ylim(y_bounds['obj value'])

    # axis[0].set_ylim([low_bound_y, high_bound_y])
    axis[0].set_xlim([low_bound_x, high_bound_x])
    if not cumulative:
        axis[0].legend([plot_name], loc='upper right')
        axis[0].savefig("plots/" + plot_name)

    objective_values= np.array(objective_values)
    objective_values= objective_values[1:]-objective_values[:-1]
    for i in range(len(objective_values)-1,0,-1):
        objective_values[i]/=objective_values[i-1]
    objective_values=np.log(np.abs(objective_values[1:]))
    axis[1].plot(range(len(objective_values)),
             objective_values)

    axis[1].set_xlabel('oracle calls ')
    axis[1].set_ylabel('log of rate (obj value)')
    axis[1].set_ylim(y_bounds['log of rate (obj value)'])
    axis[1].set_xlim([low_bound_x, high_bound_x])
    if not cumulative:
        axis[1].legend([plot_name], loc='upper right')
        axis[1].savefig("plots/" + plot_name)

def plot_log(values, y_label, plot_name,axis):
    if not cumulative:
        axis.clf()
    else:
        plot_names.append(plot_name)

    axis.plot(range(len(values)),
             values)
    axis.set_ylim(y_bounds[y_label])
    axis.set_xlim(low_bound_x,high_bound_x)
    axis.set_xlabel('iteration')
    axis.set_ylabel(y_label)
    if not cumulative:
        axis.legend([plot_name, ], loc='upper right')
        axis.savefig("plots/" + plot_name)


def test_01():
    print("running test 1...")

    gd_solver = SubProblemCubicNewton(max_iter=15, c_prime=1)
    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.1, l=2, epsilon=1e-4, n1=500, n2=500)
    thetas = ms.run(np.random.RandomState(seed=42).rand(1))
    # plot the objective value list
    plot_objective_value(oracle.objective_values, "OneDimQuad_SolverCubicNewtonMiniBatch")
    print("best parameters", thetas, '\n')


def test_02():
    print("running test 2...")

    gd_solver = SubProblemCubicNewton(max_iter=1000, c_prime=0.01)
    oracle = MeanOracle(objective_function=ThreeDimQuad(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=2000, ro=0.001, l=4, epsilon=1e-5, n1=10000, n2=10000)

    print("best parameters", ms.run(np.random.RandomState(seed=42).rand(3)), '\n')


def test_03():
    print("running test 3...")
    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = SolverSGD(oracle=oracle, max_iter=1000, n1=1, lr=0.01)
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    plot_objective_value(oracle.objective_values, "OneDimQuad_SolverSGD")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "normOfGradOneDimQuad_SolverSGD", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "normOfDiffOneDimQuad_SolverSGD",axis[3])

    print("best parameters", thetas, '\n')


def test_04():
    print("running test 4...")
    oracle = StormOracle(objective_function=OneDimQuad(), k=0.1, c_factor=300, )
    ms = SolverStorm(oracle=oracle, max_iter=1000, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    plot_objective_value(oracle.objective_values, "OneDimQuad_SolverStorm")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "normOfGradOneDimQuad_SolverStorm", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "normOfDiffOneDimQuad_SolverStorm",axis[3])

    print("best parameters", thetas, '\n')


def test_05():
    print("running test 5...")

    gd_solver = SubProblemCubicNewton(max_iter=10, c_prime=0.1)
    oracle = NormalNoiseOracle(objective_function=WLooking(), )
    ms = SolverCubicNewtonMiniBatch(sub_solver=gd_solver, oracle=oracle,
                                    max_iter=60000, ro=1, l=100, epsilon=1e-4, n1=100, n2=100)
    thetas = ms.run(np.random.RandomState(seed=45).rand(2))
    plot_objective_value(oracle.objective_values, "WLooking_SolverCubicNewtonMiniBatch")
    print("best parameters", thetas, '\n')


def test_06():
    print("running test 6...")

    oracle = NormalNoiseOracle(objective_function=WLooking(), )
    ms = SolverSGD(oracle=oracle, max_iter=60000, lr=1e-3, n1=100, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(2))

    # plot the objective value list
    plot_objective_value(oracle.objective_values, "WLooking_SolverSGD")
    plt.savefig("plots/WLooking_SolverSGD")
    print("best parameters", thetas, '\n')


def test_07():
    print("running test 7...")

    oracle = LeastSquareOracle(objective_function=OneDimQuad(), )
    ms = GradientLeastSquares(oracle=oracle, j1=1, j2=1, l=5, alpha=0.99, max_iter=1000, lr=0.01,
                              point_limit=20, use_beta=True, momentum=0.7, markov_eps=2, cut_off_eps=1e-3)
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))

    # plot the objective value list
    plot_objective_value(oracle.objective_values, "OneDimQuad_GradientLeastSquares_j1")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "normOfGradOneDimQuad_GradientLeastSquares_j1", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "normOfDiffOneDimQuad_GradientLeastSquares_j1",axis[3])
    plot_log(oracle.norm_of_beta_diff, "norm of diff (beta)", "normOfDiffOneDimQuad_GradientLeastSquares_j1",axis[4])
    # plot_log(oracle.norm_of_hessian_vec_diff, "norm of diff (Hv)", "normOfDiffOneDimQuad_GradientLeastSquares_j1",axis[4])

    print("best parameters", thetas, '\n')

cumulative = True
fig, axis = plt.subplots(5)
high_bound_x = 1000
low_bound_x = -1
high_bound_y = 3
y_bounds={"obj value":[0.5, 3],
          "log of rate (obj value)":[-7,7],
          "norm of diff (grads)":[0,1],
          "log of norm (grad)":[-10,2],
          "norm of diff (beta)":[0,3.1],
          "norm of diff (Hv)":[0,3]
          }
low_bound_y = 0

# test_01()
test_03()
test_04()
# test_05()
# test_06()
test_07()


if cumulative:
    fig.set_size_inches(12, 15)
    fig.suptitle("with_beta/alpha=0.99/pnt_limit=20/momentum=0.7/lr=0.01",ha= 'right')
    fig.legend(['OneDimQuad_SGD','OneDimQuad_Storm','OneDimQuad_GradientLeastSquare'], loc='upper right')
    plt.savefig("plots/test/" + '0', )
