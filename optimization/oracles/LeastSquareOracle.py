#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
# "StochasticOracle" is a base class that requires implementation of functions
# to return the value, gradient, symmetric sparse hessian and hessian vector product.
# The functions to get value/derivate information here are "virtual" and need to be
# implemented for specific functions.
import numpy as np

from optimization.utils.ObjectiveFunction import ObjectiveFunction
from optimization.utils.StochasticOracle import StochasticOracle


class LeastSquareOracle(StochasticOracle):

    def __init__(self, objective_function: ObjectiveFunction):
        self.generated_samples = []
        super().__init__(objective_function)

    def compute_oracle(self, x_t, **kwargs):
        objective_value = self.objective_function.forward(x_t, self.s1).mean(axis=0)
        g_t = self.objective_function.gradient(x_t, self.s1).mean(axis=0)
        return objective_value, g_t, None, None

    def update_sample(self,
                      x_t, save=True):
        super(LeastSquareOracle, self).update_sample(x_t)
        if save: self.generated_samples.append(self.s1)

    def compute_index_oracle(self, x_t, index):
        sample = self.generated_samples[index]
        objective_value = self.objective_function.forward(x_t, sample).mean(axis=0)
        g_t = self.objective_function.gradient(x_t, sample).mean(axis=0)

        return objective_value, g_t, None, None

    def log_markov_inequality(self, x_t, delta, markov_eps, markov_prob):
        print("markov prob: ", markov_prob)
        print("upperbound: ", np.exp(self.norm_of_gradients).mean() / markov_eps)
