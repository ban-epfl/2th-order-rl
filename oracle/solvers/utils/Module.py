import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):

    @abstractmethod
    def forward(self,
                x: np.ndarray,
                z: np.ndarray
                ) -> np.ndarray:
        """
        compute f(x)
        Args:
            x: np.ndarray with the length of parameters
            z: np.ndarray with the shape of (sample_num, sample_dimension)
            kwargs: dict()
        Returns:
            np.ndarray i.e [f(s1),f(s2),f(s3)...]
        """
        pass

    @abstractmethod
    def gradient(self,
                 x: np.ndarray,
                 z: np.ndarray
                 ) -> np.ndarray:
        """
        compute f'(x)
        Args:
            x: np.ndarray with the length of parameters
            z: np.ndarray with the shape of (sample_num, sample_dimension)
            kwargs: dict()
        Returns:
            np.ndarray with the shape of (sample_num, parameter_num)
        """

    @abstractmethod
    def get_param_num(self,
                      ) -> int:
        pass

    def hessian_vector(self, x_t, v, z, r):
        """
        compute f"(x).d ~ (f'(x+r.v)-f'(x-r.v))/(2*r)
        Args:
            x_t: np.ndarray with the length of parameteres
            v: np.ndarray with the length of parameteres
            samples: np.ndarray with the shape of (sample_num, sample_dimension)
            r: scaler
            kwargs: dict()
        Returns:
            np.ndarray with the shape of (sample_num, parameter_num)
        """
        return (self.gradient(x_t + r * v, z) -
                self.gradient(x_t - r * v, z)) / (2 * r)
