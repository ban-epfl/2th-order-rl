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

from optimization.utils.Module import Module
from abc import ABC, abstractmethod


class Oracle(ABC):

    def __init__(self, module: Module,
                 n1: int,
                 n2: int,
                 r=0.001):
        self.module = module
        self.r = r
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
