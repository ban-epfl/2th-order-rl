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

from typing import Tuple, Any
import numpy as np

from optimization.utils.ObjectiveFunction import ObjectiveFunction
from abc import ABC, abstractmethod


class StochasticOracle(ABC):

    def __init__(self, objective_function: ObjectiveFunction,
                 r=0.001):
        self.objective_function = objective_function
        self.r = r
        self.s1 = None
        self.s2 = None
        self.n1 = None
        self.n2 = None
        self.objective_values = []
        self.norm_of_gradients = []
        self.norm_of_grad_diff = []
        self.norm_of_hessian_vec_diff = []
        self.norm_of_beta_diff = []
        self.lastObjective_value = None

    def set_oracle_sample_size(self,
                               n1: int,
                               n2: int):
        self.n1 = n1
        self.n2 = n2

    @abstractmethod
    def compute_oracle(self,
                       x_t: np.ndarray,
                       **kwargs
                       ) -> Tuple[float, np.ndarray, Any, Any]:
        """
        compute the tuple of [gradient , hessian-vector function]
        Args:
            x_t: np.ndarray
        Returns:
            Tuple[np.ndarray, np.ndarray, function]
            the first one in the above list is F(X) as a float type
            the second one in the above list is gradient of F(X) with the length of parameter_num
            the third one in the above list is hessian function which computes the hessian-vector by getting list of samples
        """
        pass

    def update_sample(self,
                      x_t: np.ndarray,
                      save= False):
        """
        update the samples in Oracle class ( self.s1 and self.s2 )

        Nothing Returns

        """
        if self.n1 is None or self.n2 is None:
            raise ValueError('Please set oracle sample size n1 and n2!')

        # self.s1, self.s2 = self.objective_function.get_samples(self.n1, self.n2)
        # tem_s1, _ = self.objective_function.get_log_samples(50, 0)
        # if self.lastObjective_value is None:
        #     self.objective_values += [self.objective_function.forward(x_t, tem_s1).mean(axis=0)] * (self.n1 + self.n2)
        # else:
        #     self.objective_values += [self.objective_function.forward(x_t, tem_s1).mean(
        #         axis=0) - self.lastObjective_value] * (self.n1 + self.n2)

        # self.estimated_norm_gradients += [np.linalg.norm(self.objective_function.gradient(x_t, tem_s1).mean(
        # axis=0), ord=2)] * (self.n1+self.n2)

    def log_changes(self, x_t, delta, beta=None, hessian_vec=None):
        tem_s1, _ = self.objective_function.get_log_samples(100, 0)
        if beta is not None:
            self.norm_of_beta_diff.append(
                np.abs(beta - self.objective_function.true_value(x_t)))
        if hessian_vec:
            self.norm_of_hessian_vec_diff.append(
                np.linalg.norm(hessian_vec - self.objective_function.true_hessian_vector(x_t), ord=2))

        self.objective_values.append(self.objective_function.forward(x_t, tem_s1).mean(axis=0))
        self.norm_of_gradients.append(np.log(np.linalg.norm(delta, ord=2)))
        self.norm_of_grad_diff.append(np.linalg.norm(delta - self.objective_function.true_gradient(x_t), ord=2))

    def compute_index_oracle(self, x_t, index):
        raise ValueError('Not Implemented Error!')

    def log_markov_inequality(self, x_t, delta,markov_eps, markov_prob):
        raise ValueError('Not Implemented Error!')
