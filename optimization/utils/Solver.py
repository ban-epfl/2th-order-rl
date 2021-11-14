#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
#
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com

from abc import ABC, abstractmethod

import numpy as np

from optimization.oracles.StochasticOracle import StochasticOracle


class Solver(ABC):

    def __init__(self, oracle: StochasticOracle, ro, l, max_iter, epsilon):
        self.ro = ro
        self.l = l
        self.oracle = oracle
        self.max_iter = max_iter
        self.epsilon = epsilon

    @abstractmethod
    def run(self,
            x_t: np.ndarray,
            **kwargs) -> np.array:
        """
        find the best fitted parameter to minimize the objective function provided in optimization
        Args:
            x_t: np.ndarray
            kwargs: dict()
        Returns:
            np.ndarray
        """
        pass
