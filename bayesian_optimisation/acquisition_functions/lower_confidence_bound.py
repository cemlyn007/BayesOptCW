import numpy as np
import numpy.typing as npt
from gp.gaussian_process import GaussianProcess

from bayesian_optimisation.acquisition_functions.abstract_acquisition_function import (
    AcquisitionFunction,
)


class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, confidence_rate: float):
        super(LowerConfidenceBound, self).__init__()
        assert confidence_rate >= 0
        self._confidence_rate = confidence_rate

    @property
    def confidence_rate(self) -> float:
        return self._confidence_rate

    @confidence_rate.setter
    def confidence_rate(self, new_confidence_rate: float):
        assert new_confidence_rate >= 0
        self._confidence_rate = new_confidence_rate

    def _evaluate(
        self,
        gaussian_process: GaussianProcess,
        data_points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number
        of elements to evaluate
        and m is the number of variables used to calculate the objective
        function
        :return: a numpy array of shape n x 1 (or a float) representing the
        estimation of the acquisition function at each point
        """
        mean, std = gaussian_process.get_gp_mean_std(data_points)
        return -1 * (mean - np.sqrt(self.confidence_rate) * std)
