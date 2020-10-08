import numpy as np
from kernels.abstract_kernel import Kernel
from scipy.spatial import distance_matrix, minkowski_distance_p, minkowski_distance


class GaussianKernel(Kernel):
    def __init__(self,
                 log_amplitude: float,
                 log_length_scale: float,
                 log_noise_scale: float,
                 ):
        super(GaussianKernel, self).__init__(log_amplitude,
                                             log_length_scale,
                                             log_noise_scale,
                                             )

    def get_covariance_matrix(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              ) -> np.ndarray:
        """
        :param X: numpy array of size n_1 x l for which each row (x_i) is a data
                    point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data
                    point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at
                    position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        # TODO:
        if X.ndim == 1:
            X = X.reshape((len(X), -1))
        if Y.ndim == 1:
            Y = Y.reshape((len(Y), -1))

        return self.amplitude_squared * np.exp(-distance_matrix(X, Y) ** 2 / (2 * self.length_scale ** 2))

        # xnorms_2 = np.diag(X.dot(X.T)).reshape(len(X), -1)
        # ynorms_2 = np.diag(Y.dot(Y.T)).reshape(len(Y), -1)
        # xnorms_2 = xnorms_2 @ np.ones((1, ynorms_2.shape[0]))
        # ynorms_2 = np.ones((xnorms_2.shape[0], 1)) @ ynorms_2.T
        # return self.amplitude_squared * np.exp(-(xnorms_2 + ynorms_2 - 2 * X @ Y.T) / (2 * self.length_scale ** 2))

    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ) -> np.ndarray:
        return self.get_covariance_matrix(X, Y)
