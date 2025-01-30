from typing import Tuple, Union

import numpy as np
from gp.objective_function import ObjectiveFunction
from gp.parameter_category import TypeVariable
from matplotlib import pyplot as plt


class UnivariateObjectiveFunction(ObjectiveFunction):
    def __init__(
        self,
        additional_gaussian_noise_std: float = 0.0,
    ):
        super(UnivariateObjectiveFunction, self).__init__(additional_gaussian_noise_std)

    def evaluate_without_noise(
        self,
        x: Union[np.ndarray, float],
    ) -> Union[np.ndarray, float]:
        """
        Same as evaluate(data_points) but does not apply any additional noise
        to the results
        :param data_points: numpy array of dimension n x m where n is the number
        of elements to evaluate
        and m is the number of variables used to calculate the objective
        function
        :return:  a numpy array of dimension n x 1 representing all the
        evaluations for all the n elements.
        """
        result = np.sin(x) + np.sin((10.0 / 3.0) * x)
        return result

    @property
    def dataset_bounds(self) -> Tuple[Tuple[Tuple[float, float], TypeVariable], ...]:
        """
        Defines the bounds and the types of variables for the objective function

        Example:
        if dataset_bounds is equal to
        (
        ((1, 2), TypeVariable.REAL),
        ((5, 10), TypeVariable.INTEGER),
        )
        then it means the objective function depends on 2 variables:
        - the first one is a real number between 1 and 2
        - the second one is an integer between 5 (included) and 10 (excluded)
        """

        return (((2.0, 8.0), TypeVariable.REAL),)

    def plot(self, list_number_points_per_axis):
        mesh_grid = self.get_mesh_grid(list_number_points_per_axis)

        evaluations = self.evaluate(mesh_grid[0])
        if not self._additional_gaussian_noise_std:
            plt.plot(mesh_grid[0].flatten(), evaluations.flatten())
        else:
            plt.scatter(mesh_grid[0].flatten(), evaluations.flatten())
        plt.show()
