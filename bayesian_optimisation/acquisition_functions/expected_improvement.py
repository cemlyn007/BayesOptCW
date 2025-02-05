import numpy as np
import numpy.typing as npt
import scipy.stats
from gp.gaussian_process import GaussianProcess

from bayesian_optimisation.acquisition_functions.abstract_acquisition_function import (
    AcquisitionFunction,
)


class ExpectedImprovement(AcquisitionFunction):
    def _evaluate(
        self, gaussian_process: GaussianProcess, data_points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the
        number of elements to evaluate and m is the number of variables used to
        calculate the objective function
        :return: a numpy array of shape n (or a float) representing the
        estimation of the acquisition function at each point
        """
        mean, std = gaussian_process.get_gp_mean_std(data_points)
        function_values = gaussian_process.array_objective_function_values
        gamma = (np.min(function_values) - mean) / std
        return std * (gamma * scipy.stats.norm.cdf(gamma) + scipy.stats.norm.pdf(gamma))
