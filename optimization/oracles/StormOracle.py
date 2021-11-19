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


class StormOracle(StochasticOracle):

    def __init__(self, objective_function: ObjectiveFunction, n1=1, k=0.1, w=0.1, c_factor=0.01, ):
        self.k = k
        self.w = w
        self.c_factor = c_factor
        self.grads=[]
        self.sqr_grads_norms=0
        self.d=None
        self.eta=k/(w**(1/3))

        super().__init__(objective_function, n1, 0)

    def compute_oracle(self, x_t, **kwargs):
        objective_value = self.objective_function.forward(x_t, self.s1).mean(axis=0)
        gradient = self.objective_function.gradient(x_t, self.s1).mean(axis=0).reshape(-1)

        # Storing all gradients in a list
        self.grads.append(gradient)

        # Calculating and storing ∑G^2in sqrgradnorm
        self.sqr_grads_norms += np.power(np.linalg.norm(gradient, ord=2), 2)

        # Calculating and storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
        if self.d is None:
            # Updating learning rate('η' in paper)
            power = 1.0 / 3.0
            scaling = np.power((0.1 + self.sqr_grads_norms), power)
            self.eta = self.k / (float)(scaling)
            # Storing the momentum term
            self.d = self.grads[-1]
        else:
            # Updating learning rate('η' in paper)
            # Calculating 'a' mentioned as a=cη^2 in paper(denoted 'c' as factor here)
            a = min(self.c_factor * self.eta ** 2.0, 1.0)
            power = 1.0 / 3.0
            scaling = np.power((0.1 + self.sqr_grads_norms), power)
            self.eta = self.k / (float)(scaling)
            # Storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
            self.d = self.grads[-1] + (1 - a) * (self.d - self.grads[-2])

        return objective_value, self.d, None, self.eta
