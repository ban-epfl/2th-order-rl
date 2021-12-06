#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# This implements the least squares estimator for the gradient. Possible field of
# this class are the past iterates of the master problem, the convex combination 
# parameter, the number of samples and other precision parameters.
#
import quadprog
import numpy as np

from optimization.oracles.LeastSquareOracle import LeastSquareOracle
from optimization.utils.Solver import Solver


# from sklearn import datasets, linear_model


class GradientLeastSquares(Solver):

    def __init__(self, oracle: LeastSquareOracle, lr=0.001, l=1, j1=100, j2=100, alpha=0.1, max_iter=10000,
                 point_limit=300, use_beta=False, momentum=0.9, markov_eps=None, cut_off_eps=None):
        self.j2 = j2
        self.alpha = alpha
        self.x_js = []
        self.point_limit = point_limit
        self.use_beta = use_beta
        self.momentum = momentum
        self.markov_eps = markov_eps
        self.cut_off_eps = cut_off_eps
        super().__init__(oracle, None, l, max_iter, None, lr, 1, 0)

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = np.matmul(X, theta)
        return 1 / (2 * m) * np.linalg.norm(h - y, ord=2) ** 2

    def least_squares_SGD(self, c, A, theta, num_iters=100, gamma=0.001, ):
        m = c.shape[0]
        models = []
        for iter in range(num_iters):
            random_idx = np.random.randint(m)
            # update weights
            theta = theta - gamma * np.transpose(A) * (np.matmul(A, theta) - c)
            models.append(np.copy(theta))
            # best_model_idx = np.argmin(val_loss)
        return models[-1]

    def get_random_x_js(self, x_t):
        """
        Args:
            x_t: np.ndarray

        Returns:
         a matrix with the shape of (self.j1, x_t.shape[0]) OR (self.j1, param_num)

        """
        return x_t + np.random.normal(0, 0.01, (100, x_t.shape[0]))

    def estimate_hessian_vector(self, x_js, x_t, v):
        """
        Args:
            np.ndarray :param x_js: last point in the method
            np.ndarray :param x_t: current point we are going to find its gradient
            np.ndarray :param v:   the vector which is going to compute Hv ( hessian vector )
        Returns:
            np.ndarray :return:  Hessian vector ( Hv )
        """
        x_js_minus = x_js - x_t
        x_js_minus = np.c_[np.ones(x_js.shape[0]), x_js_minus]
        b = np.zeros(x_js.shape[0])
        for j in range(x_js.shape[0]):
            _, g_j, _, _ = self.oracle.compute_index_oracle(x_js[j], j)
            b[j] = np.dot(g_j, v)
        result = np.linalg.lstsq(x_js_minus, b, rcond=None)[0]
        return result[1:]

    def run(self, x_t, **kwargs):
        print("GradientLeastSquares optimizing... ")
        self.oracle.update_sample(x_t, save=False)
        _, g_t_initial, _, _ = self.oracle.compute_oracle(x_t, )
        self.x_js.append(x_t - self.lr*g_t_initial)
        self.oracle.update_sample(x_t, )
        inactives_l=[]
        markov_counter = 0
        # initiate nestrov momentum velocity
        v_t = np.zeros(x_t.shape)
        for i in range(self.max_iter):
            # to save original x_t for nestrov update
            x_t_original = x_t
            x_t = x_t - self.momentum * v_t
            # x_js=self.get_random_x_js(x_t)
            x_js = np.array(self.x_js)
            self.x_js.append(x_t)
            # if last points are going to increase the limit, cut them out
            if x_js.shape[0] > self.point_limit:
                x_js = x_js[x_js.shape[0] - self.point_limit:]
            # solve least square problem
            # delta = self.find_least_square(x_js, x_t,)
            # solve qd problem
            delta, inactives = self.solve_qp(x_js, x_t, )
            inactives_l.append(inactives)
            # estimate hessian vector
            hessian_vec = None
            # hessian_vec = self.estimate_hessian_vector(x_js,
            #                                            x_t,
            #                                            np.random.RandomState(seed=2).rand(x_t.shape[0]))

            if self.use_beta:
                self.oracle.log_changes(x_t, delta[1:], delta[0], hessian_vec=hessian_vec)
                delta = delta[1:]
            else:
                self.oracle.log_changes(x_t, delta, hessian_vec=hessian_vec)

            norm_of_delta = np.linalg.norm(delta, ord=2)
            if self.markov_eps and norm_of_delta > self.markov_eps:
                markov_counter += 1

            if self.cut_off_eps and norm_of_delta < self.cut_off_eps:
                self.oracle.log_markov_inequality(x_t, norm_of_delta, markov_eps=self.markov_eps,
                                                  markov_prob=markov_counter / (i + 1))
                break

            self.oracle.update_sample(x_t, )
            # NAG: Nesterov accelerated gradient
            v_t = self.momentum * v_t + self.lr * delta
            x_t = x_t_original - v_t

        return x_t, inactives_l

    def solve_qp(self, x_js, x_t, ):
        x_js_minus = x_js - x_t
        # in case of estimating beta, so we need to increase the dim if x_js
        if self.use_beta: x_js_minus = np.c_[np.ones(x_js.shape[0]), x_js_minus]
        G_matrix = np.identity(x_js.shape[0] + (1 if self.use_beta else 0) + x_t.shape[0])
        for i in range(G_matrix.shape[0]):
            if i < x_js.shape[0]:
                G_matrix[i][i] *= self.alpha / x_js.shape[0]
            else:
                G_matrix[i][i] *= (1.0 - self.alpha) / self.j2

        G = 2 * G_matrix
        C = np.zeros((2 * x_js.shape[0], G.shape[0]))
        h = np.zeros((2 * x_js.shape[0],))
        a = np.zeros(G.shape[0])
        # compute mean of gradient estimate of the right hand side of the equation Ad=c
        gradient_sum = np.zeros(x_t.shape)
        for j in range(self.j2):
            self.oracle.update_sample(x_t, save=False)
            objective_value_x_t, g_t, _, _ = self.oracle.compute_oracle(x_t, )
            gradient_sum += g_t
        # in case of estimating beta, so we need to increase the dim if gradient_sum
        if self.use_beta:
            gradient_sum = np.concatenate([[0], gradient_sum])
        a[-gradient_sum.shape[0]:] = -2 * (1.0 - self.alpha) / self.j2 * gradient_sum
        # build constraints
        j = 0
        for i in range(C.shape[0]):
            objective_value_x_j, _, _, _ = self.oracle.compute_index_oracle(x_js[j], j)
            objective_value_x_t, _, _, _ = self.oracle.compute_index_oracle(x_t, j)
            if self.use_beta: objective_value_x_t = 0
            if i % 2 == 0:
                C[i][j] = -1
                C[i][-x_js_minus[j].shape[0]:] = -x_js_minus[j]
                h[i] = self.l / 2 * np.linalg.norm(x_js[j] - x_t, ord=2) ** 2 - (
                            objective_value_x_j - objective_value_x_t)
            else:
                C[i][j] = 1
                C[i][-x_js_minus[j].shape[0]:] = x_js_minus[j]
                h[i] = self.l / 2 * np.linalg.norm(x_js[j] - x_t, ord=2) ** 2 + (
                            objective_value_x_j - objective_value_x_t)
                j += 1

        # sol=solvers.qp(Q, None, G, h, None, None)
        sol = quadprog.solve_qp(G, -a, -C.T, -h, 0)
        # print("********************************")
        # # print(sol[0][:x_js.shape[0]])
        # print(sum(sol[0][:x_js.shape[0]]==0))
        return sol[0][x_js.shape[0]:], sum(sol[0][:x_js.shape[0]]==0)

    def find_least_square(self, x_js, x_t, ):
        x_js_minus = x_js - x_t
        b = 0
        # in case of estimating beta, so we need to increase the dim if x_js
        if self.use_beta: x_js_minus = np.c_[np.ones(x_js.shape[0]), x_js_minus]
        A = self.alpha / x_js.shape[0] * np.matmul(x_js_minus.T, x_js_minus) + (1.0 - self.alpha) * np.identity(
            x_t.shape[0] + (1 if self.use_beta else 0))
        # compute lipschitz part of the right hand side of the equation Ad=c
        for j in range(x_js.shape[0]):
            # self.oracle.update_sample(x_t)
            objective_value_x_j, _, _, _ = self.oracle.compute_index_oracle(x_js[j], j)
            objective_value_x_t, _, _, _ = self.oracle.compute_index_oracle(x_t, j)
            if self.use_beta: objective_value_x_t = 0
            b += (self.alpha / x_js.shape[0]) * (
                    objective_value_x_j - objective_value_x_t - self.l / 2 * np.linalg.norm(x_js[j] - x_t,
                                                                                            ord=2) ** 2) * \
                 x_js_minus[j]

        # compute mean of gradient estimate of the right hand side of the equation Ad=c
        gradient_sum = np.zeros(x_t.shape)
        for j in range(self.j2):
            self.oracle.update_sample(x_t, )
            objective_value_x_t, g_t, _, _ = self.oracle.compute_oracle(x_t, )
            gradient_sum += g_t

        # in case of estimating beta, so we need to increase the dim if gradient_sum
        if self.use_beta:
            gradient_sum = np.concatenate([[0], gradient_sum])
        b += (1.0 - self.alpha) / self.j2 * gradient_sum

        return np.linalg.lstsq(A / x_t.shape[0], b / x_t.shape[0], rcond=None)[0]
