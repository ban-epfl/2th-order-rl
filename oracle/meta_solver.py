import numpy as np

from oracle.solvers.solver_GD import SolverGD


class MetaSolver:

    def __init__(self, sub_solver, obj_func, obj_func_prime,
                 ro=0.1, l=0.5, n1=20, n2=20, r=0.001):
        self.ro = ro
        self.l = l
        self.n1 = n1
        self.n2 = n2
        self.r = r
        self.sub_solver = sub_solver
        self.obj_func = obj_func
        self.obj_func_prime = obj_func_prime

    def hessian_vector(self, x_t, v, s2_i):

        return (self.obj_func_prime(x_t + self.r * v, s2_i) -
                self.obj_func_prime(x_t - self.r * v, s2_i)) / (2 * self.r)

    def get_oracle(self,x_t):

        s1 = np.random.normal(0,1,(self.n1, 3))
        g_t = self.obj_func_prime(x_t, s1).reshape((self.n1,1)).mean(axis=0)
        s2 = np.random.normal(0,1,(self.n2, 3))
        B_t = lambda v: 1 / self.n2 * sum([self.hessian_vector(x_t, v, s2[i:i+1]) for i in range(self.n2)])

        return g_t, B_t

    def run(self, x_t, max_iter=1000, epsilon=0.001):

        for i in range(max_iter):
            g_t, B_t = self.get_oracle(x_t)
            delta, delta_m = self.sub_solver.solve(g_t, B_t, epsilon)
            x_t1 = x_t + delta
            if delta_m >= -(epsilon ** 3 / self.ro) ** 0.5 / 100:
                delta = self.sub_solver.final_solve(g_t, B_t, epsilon)
                return x_t + delta
            x_t = x_t1

        return x_t


def obj_func(x, z):
    return z[:, 0] * x ** 2 + z[:, 1] * x + z[:, 2]


def obj_func_prime(x, z):
    return 2 * z[:, 0] * x + z[:, 1]


gd_solver=SolverGD(0.1,0.5,100000,0.01)
ms=MetaSolver(gd_solver, obj_func, obj_func_prime)
print("best parameters", ms.run(np.random.rand(1)[0]))