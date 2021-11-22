import numpy as np

from optimization.utils.ObjectiveFunction import ObjectiveFunction


class WLooking(ObjectiveFunction):

    def __init__(self):
        self.L = 5
        self.e = 0.01

    def get_sample_dim(self) -> int:
        return 2

    def forward(self, x, z):
        L = self.L
        e = self.e
        sqrt_e = e ** 0.5
        w_output = None
        if x[0] <= -L * sqrt_e:
            w_output = sqrt_e * (x[0] + (L + 1) * sqrt_e) ** 2 - 1 / 3 * (x[0] + (L + 1) * sqrt_e) ** 3 - 1 / 3 * (
                    3 * L + 1) * e ** 1.5
        elif -L * sqrt_e < x[0] <= -sqrt_e:
            w_output = e * x[0] + e ** 1.5 / 3
        elif -sqrt_e < x[0] <= 0:
            w_output = -sqrt_e * x[0] ** 2 - x[0] ** 3 / 3
        elif 0 < x[0] <= sqrt_e:
            w_output = -sqrt_e * x[0] ** 2 + x[0] ** 3 / 3
        elif sqrt_e < x[0] <= L * sqrt_e:
            w_output = -e * x[0] + e ** 1.5 / 3
        elif L * sqrt_e <= x[0]:
            w_output = sqrt_e * (x[0] - (L + 1) * sqrt_e) ** 2 + 1 / 3 * (x[0] - (L + 1) * sqrt_e) ** 3 - 1 / 3 * (
                    3 * L + 1) * e ** 1.5

        return np.array([w_output + 10 * x[1] ** 2] * z.shape[0])

    def gradient(self, x, z):
        L = self.L
        e = self.e
        sqrt_e = e ** 0.5
        w_output = None
        if x[0] <= -L * sqrt_e:
            w_output = 2 * sqrt_e * (x[0] + (L + 1) * sqrt_e) - (x[0] + (L + 1) * sqrt_e) ** 2
        elif -L * sqrt_e < x[0] <= -sqrt_e:
            w_output = e
        elif -sqrt_e < x[0] <= 0:
            w_output = -2 * sqrt_e * x[0] - x[0] ** 2
        elif 0 < x[0] <= sqrt_e:
            w_output = -2 * sqrt_e * x[0] + x[0] ** 2
        elif sqrt_e < x[0] <= L * sqrt_e:
            w_output = -e + e ** 1.5 / 3
        elif L * sqrt_e <= x[0]:
            w_output = 2 * sqrt_e * (x[0] - (L + 1) * sqrt_e) + (x[0] - (L + 1) * sqrt_e) ** 2

        return np.tile([w_output, 20 * x[1]], (z.shape[0], 1))

    def hessian_vector(self, x, v, z, r):
        L = self.L
        e = self.e
        sqrt_e = e ** 0.5
        if x[0] <= -L * sqrt_e:
            w_output = 2 * sqrt_e - 2 * (x[0] + (L + 1) * sqrt_e)
        elif -sqrt_e < x[0] <= 0:
            w_output = -2 * sqrt_e - 2 * x[0]
        elif 0 < x[0] <= sqrt_e:
            w_output = -2 * sqrt_e + 2 * x[0]
        elif L * sqrt_e <= x[0]:
            w_output = 2 * sqrt_e + 2 * (x[0] - (L + 1) * sqrt_e)
        else:
            w_output = 0
        hessian = np.tile([[w_output, 0], [0, 20]], (z.shape[0], 1, 1))
        return np.matmul(hessian, v.reshape(2, 1)).squeeze()


class OneDimQuad(ObjectiveFunction):

    def get_sample_dim(self) -> int:
        return 3

    def forward(self, x, z):
        return z[:, 0] * x ** 2 + z[:, 1] * x + z[:, 2]

    def gradient(self, x, z):
        return np.expand_dims(2 * z[:, 0] * x + z[:, 1], axis=1)

    # def hessian_vector(self, x_t, v, z, r):
    #     return np.matmul(np.expand_dims(2 * z[:, 0], axis=1), np.expand_dims(v, axis=1))

    def get_samples(self,
                    n1: int,
                    n2: int):

        s1 = np.random.normal(1, 0.1, (n1, self.get_sample_dim()))
        s2 = np.random.normal(1, 0.1, (n2, self.get_sample_dim()))
        return s1, s2

    def get_log_samples(self,
                        n1: int,
                        n2: int):

        self.log_s1 = np.random.RandomState(seed=56).normal(1, 2, (n1, self.get_sample_dim()))
        self.log_s2 = np.random.RandomState(seed=56).normal(1, 2, (n2, self.get_sample_dim()))
        return self.log_s1, self.log_s2


class ThreeDimQuad(ObjectiveFunction):

    def get_sample_dim(self) -> int:
        return 3

    def forward(self, x, z):
        return np.matmul(z[:, 0].reshape(-1, 1), (x ** 2).reshape(1, -1)).sum(axis=1) + \
               np.matmul(z[:, 1].reshape(-1, 1), x.reshape(1, -1)).sum(axis=1) + \
               z[:, 2]

    def gradient(self, x, z):
        return np.expand_dims(np.matmul(2 * z[:, 0].reshape(-1, 1), x.reshape(1, -1)).sum(axis=1) + z[:, 1], axis=1)

    def hessian_vector(self, x_t, v, z, r):
        return np.matmul(np.expand_dims(2 * z[:, 0], axis=1), np.expand_dims(v, axis=1))
