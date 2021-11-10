import numpy as np


class SolverGD:

    def __init__(self, ro, l, max_iter, c_prime):
        self.ro = ro
        self.l = l
        self.max_iter = max_iter
        self.c_prime = c_prime

    def solve(self, g_t, B_t, epsilon):
        g_t_norm = np.linalg.norm(g_t, ord=2)
        if g_t_norm >= self.l ** 2 / self.ro:

            g_t_dot_bg_t = np.matmul(g_t, B_t(g_t)) / (self.ro * (g_t_norm ** 2))
            R_c = -g_t_dot_bg_t + (g_t_dot_bg_t ** 2 + 2 * g_t_norm / self.ro) ** 0.5
            delta = -R_c * g_t / g_t_norm

        else:
            delta, sigma, zigma = np.zeros(1), self.c_prime * (epsilon * self.ro) ** 0.5 / self.l, 1 / (2 * self.l)
            khi = np.random.uniform(0, 1)
            g_tilda = g_t + sigma * khi
            for i in range(self.max_iter):
                delta -= zigma * (g_tilda + B_t(delta) + self.ro / 2 * np.linalg.norm(delta, ord=2) * delta)
        delta_m = np.matmul(g_t, delta) + 0.5 * np.matmul(delta, B_t(delta)) + self.ro / 6 * np.linalg.norm(delta,
                                                                                                            ord=3)

        return delta, delta_m

    def final_solve(self, g_t, B_t, epsilon):
        delta, g_m, zigma = 0, g_t, 1 / (2 * self.l)
        while np.linalg.norm(g_m, ord=2) > epsilon:
            delta = delta - zigma * g_m
            g_m = g_t + B_t(delta) + self.ro / 2 * np.linalg.norm(delta, ord=2) * delta

        return delta
