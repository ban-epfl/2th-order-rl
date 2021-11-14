from typing import Tuple

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
    def get_sample_dim(self,
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

    def get_samples(self,
                    n1: int,
                    n2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
            get batch samples
            Args:
                n1: batch size for gradient
                n2: batch size for hessian

            Returns:
                Tuple[np.ndarray, np.ndarray]
        """
        s1 = np.random.normal(1, 2, (n1, self.get_sample_dim()))
        s2 = np.random.normal(1, 2, (n2, self.get_sample_dim()))
        return s1, s2