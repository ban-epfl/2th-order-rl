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
from optimization.solvers.SolverSGDHess1 import SolverSGDHess1
from optimization.test_modules import WLooking, OneDimQuad, ThreeDimQuad, HighDimQuad
from optimization.solvers.SolverSGD import SolverSGD
from optimization.solvers.SolverStorm import SolverStorm
from optimization.solvers.SolverCubicNewtonMiniBatch import SolverCubicNewtonMiniBatch
from optimization.solvers.SubProblemCubicNewton import SubProblemCubicNewton

plot_names = []
file_name = "-1"
save_path = "plots/"


def plot_objective_value(objective_values, plot_name):

    axis[0].plot(range(len(objective_values)),
                 objective_values)

    axis[0].set_xlabel('oracle calls ')
    axis[0].set_ylabel('obj value')
    axis[0].set_ylim(y_bounds['obj value'])

    # axis[0].set_ylim([low_bound_y, high_bound_y])
    axis[0].set_xlim([low_bound_x, high_bound_x])
    if not cumulative:
        axis[0].legend([plot_name], loc='upper right')
        axis[0].figure.savefig("plots/" + plot_name)
    np.savetxt(save_path + "data/" + plot_name + "_obj_" + file_name + ".csv", objective_values, delimiter=",")
    objective_values = np.array(objective_values)
    objective_values = objective_values[1:] - objective_values[:-1]
    for i in range(len(objective_values) - 1, 0, -1):
        objective_values[i] /= objective_values[i - 1]
    objective_values = np.log(np.abs(objective_values[1:]))
    axis[1].plot(range(len(objective_values)),
                 objective_values)

    axis[1].set_xlabel('oracle calls ')
    axis[1].set_ylabel('log of rate (obj value)')
    axis[1].set_ylim(y_bounds['log of rate (obj value)'])
    axis[1].set_xlim([low_bound_x, high_bound_x])
    if not cumulative:
        axis[1].legend([plot_name], loc='upper right')
        axis[1].figure.savefig("plots/" + plot_name)


def plot_log(values, y_label, plot_name, axis, var=False):

    axis.plot(range(len(values)),
              values)
    axis.set_ylim(y_bounds[y_label])
    axis.set_xlim(low_bound_x, high_bound_x)
    axis.set_xlabel('iteration')
    axis.set_ylabel(y_label)
    if "norm of diff (grads)" == y_label:
        var_labels.append("std " + plot_name + " : " + "{:.4f}".format(np.std(values[600:high_bound_x])))

    if not cumulative:
        axis.legend([plot_name, ], loc='upper right')
        axis.figure.savefig("plots/" + plot_name)
    np.savetxt(save_path + "data/" + plot_name + "_" + y_label + "_" + file_name + ".csv", values, delimiter=",")


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
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "SGD", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "SGD", axis[3])

    print("best parameters", thetas, '\n')


def test_04():
    print("running test 4...")
    oracle = StormOracle(objective_function=OneDimQuad(), k=0.05, c_factor=100, )
    ms = SolverStorm(oracle=oracle, max_iter=1000, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    plot_objective_value(oracle.objective_values, "SolverStorm")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "Storm", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "Storm", axis[3])

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
    ms = GradientLeastSquares(oracle=oracle, j2=1, l=lipschitz, alpha=1, max_iter=1000, lr=lr,
                              point_limit=pnt_limit, use_beta=False, momentum=momentum, markov_eps=None,
                              cut_off_eps=None)
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))

    # plot the objective value list
    plot_objective_value(oracle.objective_values, "OneDimQuad_GradientLeastSquares")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "GradientLeastSquares", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "GradientLeastSquares", axis[3])
    # plot_log(oracle.norm_of_beta_diff, "norm of diff (beta)", "GradientLeastSquares", axis[4]) plot_log(
    # oracle.norm_of_hessian_vec_diff, "norm of diff (Hv)", "normOfDiffOneDimQuad_GradientLeastSquares_j1",axis[4])

    print("best parameters", thetas, '\n')


def test_08():
    print("running test 8...")
    oracle = MeanOracle(objective_function=HighDimQuad(quad_dim), )
    ms = SolverSGD(oracle=oracle, max_iter=1000, n1=1, lr=0.01)
    thetas = ms.run(np.random.RandomState(seed=45).rand(quad_dim))
    plot_objective_value(oracle.objective_values, "OneDimQuad_SolverSGD")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "SGD", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "SGD", axis[3])

    print("best parameters", thetas, '\n')


def test_09():
    print("running test 9...")
    oracle = StormOracle(objective_function=HighDimQuad(quad_dim), k=0.7, c_factor=1, )
    ms = SolverStorm(oracle=oracle, max_iter=1000, )
    thetas = ms.run(np.random.RandomState(seed=45).rand(quad_dim))
    plot_objective_value(oracle.objective_values, "Storm")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "Storm", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "Storm", axis[3])

    print("best parameters", thetas, '\n')


def test_10():
    print("running test 10...")

    oracle = LeastSquareOracle(objective_function=HighDimQuad(quad_dim), )
    ms = GradientLeastSquares(oracle=oracle, j2=1, l=lipschitz, alpha=alpha, max_iter=1000, lr=lr,
                              point_limit=pnt_limit, use_beta=False, momentum=momentum, markov_eps=None, cut_off_eps=None)
    thetas, inactives = ms.run(np.random.RandomState(seed=45).rand(quad_dim))

    # plot the objective value list
    plot_objective_value(oracle.objective_values, "GradientLeastSquares")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "GradientLeastSquares", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "GradientLeastSquares", axis[3])
    plot_log(inactives, "inactives", "GradientLeastSquares", axis[4])
    # plot_log(oracle.norm_of_beta_diff, "norm of diff (beta)", "GradientLeastSquares", axis[4])
    # plot_log(oracle.norm_of_hessian_vec_diff, "norm of diff (Hv)", "normOfDiffOneDimQuad_GradientLeastSquares_j1",
    # axis[4])

    print("best parameters", thetas, '\n')



def test_11():
    print("running test 11...")
    oracle = MeanOracle(objective_function=OneDimQuad(), )
    ms = SolverSGDHess1(oracle=oracle, max_iter=1000, n1=1, n2=1, lr=0.01, G=100, momentum=0.9)
    thetas = ms.run(np.random.RandomState(seed=45).rand(1))
    plot_objective_value(oracle.objective_values, "SGDHess")
    plot_log(oracle.norm_of_gradients, "log of norm (grad)", "SGDHess", axis[2])
    plot_log(oracle.norm_of_grad_diff, "norm of diff (grads)", "SGDHess", axis[3])

    print("best parameters", thetas, '\n')


cumulative = True
fig, axis = plt.subplots(5)
var_labels = []
high_bound_x = 1000
low_bound_x = -1
high_bound_y = 3
low_bound_y = 0




pnt_limit = 100
momentum = 0.9
lr = 0.001
lipschitz = 4
quad_dim = 10
alpha = 0.9999999999999999


y_bounds = {"obj value": [.5*quad_dim, 2.5*quad_dim],
            "log of rate (obj value)": [-7, 7],
            "norm of diff (grads)": [0, 10],
            "log of norm (grad)": [-8, 10],
            "norm of diff (beta)": [0, 3.1],
            "norm of diff (Hv)": [0, 6],
            "inactives": [0, pnt_limit]
            }

# test_01()
# test_03()
# test_04()
test_05()
# test_06()
# test_07()
# test_08()
# test_09()
# test_10()
# test_11()

if cumulative:
    fig.set_size_inches(12, 15)
    axis[3].legend(var_labels, loc='upper right')
    fig.suptitle("without_beta/pnt_limit={}/mom={:.2f}/lr={:.4f}/\n lpchz={}/ quad_dim={}\n alpha={}".format(pnt_limit,
                                                                                                       momentum,
                                                                                                       lr,
                                                                                                       lipschitz,
                                                                                                       quad_dim,
                                                                                                        alpha),
                 ha='right')
    fig.legend(['Quad_SGD', 'Quad_Storm', 'Quad_GradientLeastSquare', ], loc='upper right')
    plt.savefig(save_path + file_name, )

