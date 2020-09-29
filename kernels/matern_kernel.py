import numpy as np

from kernels.abstract_kernel import Kernel


class MaternKernel(Kernel):
    def get_covariance_matrix(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X: numpy array of size n_1 x l for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        # TODO
        if X.ndim == 1:
            X = X.reshape((len(X), -1))
        if Y.ndim == 1:
            Y = Y.reshape((len(Y), -1))

        xnorms_2 = np.diag(X.dot(X.T)).reshape(len(X), -1)
        ynorms_2 = np.diag(Y.dot(Y.T)).reshape(len(Y), -1)
        xnorms_2 = xnorms_2 @ np.ones((1, ynorms_2.shape[0]))
        ynorms_2 = np.ones((xnorms_2.shape[0], 1)) @ ynorms_2.T
        tmp_calc = np.sqrt(3 * (xnorms_2+ynorms_2-2*X@Y.T))/self.length_scale
        return self.amplitude_squared * (1 + tmp_calc) * np.exp(-tmp_calc)
