#
# second order methods in machine learning and reinforcement learning
#
# date: Nov/2021
# 
# author: pedro.borges.melo@gmail.com
# author: mohammadsakh@gmail.com
#
from typing import Any, Tuple
import numpy as np


class SubProblemCubicNewton:

    def __init__(self, max_iter, c_prime):
        self.max_iter = max_iter
        self.c_prime = c_prime

    def solve(self, g_t: np.ndarray,
              B_t: Any,
              epsilon: float,
              ro: float,
              l: float) -> Tuple[np.ndarray, float]:
        """
        solve the sub problem with gradient decent
        Args:
            g_t: np.ndarray,
            B_t: Any,
            epsilon: float,
            ro: float,
            l: float
        Returns:
            Tuple[np.ndarray, float]
        """
        g_t_norm = np.linalg.norm(g_t, ord=2)
        if g_t_norm >= l ** 2 / ro:
            g_t_dot_bg_t = np.matmul(g_t, B_t(g_t)) / (ro * (g_t_norm ** 2))
            R_c = -g_t_dot_bg_t + (g_t_dot_bg_t ** 2 + 2 * g_t_norm / ro) ** 0.5
            delta = -R_c * g_t / g_t_norm

        else:
            delta, sigma, eta = np.zeros(g_t.shape[0]), self.c_prime * (epsilon * ro) ** 0.5 / l, 1 / (20 * l)
            khi = np.random.uniform(0, 1, g_t.shape)
            g_tilda = g_t + sigma * khi
            for i in range(self.max_iter):
                delta -= eta * (g_tilda + eta * B_t(delta) + ro / 2 * np.linalg.norm(delta, ord=2) * delta)
        delta_m = np.dot(g_t, delta) + 0.5 * np.dot(delta, B_t(delta)) + ro / 6 * np.linalg.norm(delta, ord=2) ** 3

        return delta, delta_m

    def final_solve(self, g_t: np.ndarray,
                    B_t: Any,
                    epsilon: float,
                    ro: float,
                    l: float) -> np.ndarray:

        """
        solve the sub problem with gradient decent
        Args:
            g_t: np.ndarray,
            B_t: Any,
            epsilon: float,
            ro: float,
            l: float
        Returns:
            np.ndarray ( step amount to next param-value )
        """
        delta, g_m, eta = np.zeros(g_t.shape[0]), g_t, 1 / (20 * l)
        while np.linalg.norm(g_m, ord=2) > epsilon / 2:
            delta = delta - eta * g_m
            g_m = g_t + B_t(delta) + ro / 2 * np.linalg.norm(delta, ord=2) * delta

        return delta
