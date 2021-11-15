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

from optimization.utils.Module import Module
from optimization.utils.Oracle import Oracle


class StormOracle(Oracle):

    def __init__(self, module: Module, n1: int, n2: int, k=0.1, w=0.1, c_factor=0.01, default_a=0.1):
        self.k = k
        self.w = w
        self.c_factor = c_factor
        self.default_a = default_a
        super().__init__(module, n1, n2)

    def compute_oracle(self, x_t, **kwargs):
        s1, s2 = self.module.get_samples(1, 1)
        gradient = self.module.gradient(x_t, s1)

        # Storing all gradients in a list
        if "grads" in self.state:
            self.state["grads"].append(gradient)
        else:
            self.state["grads"] = [gradient]

        # Calculating and storing ∑G^2in sqrgradnorm
        if "sqr_grad_norm" in self.state:
            self.state["sqr_grads_norms"] += np.power(np.linalg.norm(gradient, ord=2), 2)
        else:
            self.state["sqr_grads_norms"] = np.power(np.linalg.norm(gradient, ord=2), 2)

        # Calculating and storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
        if "d" not in self.state:
            # Updating learning rate('η' in paper) if lr is in kwargs
            if "lr" in kwargs:
                power = 1.0 / 3.0
                scaling = np.power((0.1 + self.state["sqr_grads_norms"]), power)
                kwargs["lr"] = self.k / (float)(scaling)
            # Storing the momentum term
            self.state["d"] = self.state["d"][-1]
        else:
            # Updating learning rate('η' in paper) if lr is in kwargs
            if "lr" in kwargs:
                # Calculating 'a' mentioned as a=cη^2 in paper(denoted 'c' as factor here)
                a = min(self.c_factor * kwargs["lr"] ** 2.0, 1.0)
                power = 1.0 / 3.0
                scaling = np.power((0.1 + self.state["sqr_grads_norms"]), power)
                kwargs["lr"] = self.k / (float)(scaling)
            else:
                a = self.default_a
            # Storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
            self.state["d"] = self.state["grads"][-1] + (1 - a) * (self.state["d"] - self.state["grads"][-2])

        return self.state["d"], None, kwargs
