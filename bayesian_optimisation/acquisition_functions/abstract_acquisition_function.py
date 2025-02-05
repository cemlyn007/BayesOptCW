import abc
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from gp.gaussian_process import GaussianProcess
from gp.objective_function import ObjectiveFunction
from scipy import optimize


class AcquisitionFunction(metaclass=abc.ABCMeta):
    def compute_arg_max(
        self,
        gaussian_process: GaussianProcess,
        objective_function: ObjectiveFunction,
    ) -> npt.NDArray[np.float64]:
        """
        :param gaussian_process:
        :param objective_function:
        :return: The computed arg_min of the acquisition of the acquisition
        function according to:
        - the predictions (mean and std) of the gaussian process
        - the boundaries of the objective function
        The result is a flattened numpy array.
        """

        def evaluate_gaussian_process(data_point: npt.NDArray[np.float64]) -> float:
            return (
                -1 * self.evaluate(gaussian_process, objective_function, data_point)
            ).item()

        boundaries = tuple(map(itemgetter(0), objective_function.dataset_bounds))

        if len(boundaries) == 1:
            (data_points,) = objective_function.get_mesh_grid([100])
            data_points = data_points.reshape((-1, 1))
            evaluations = self.evaluate(
                gaussian_process, objective_function, data_points
            )
            return data_points[np.argmax(evaluations)]
        elif len(boundaries) == 2:
            data_points, yy = objective_function.get_mesh_grid([100, 100])
            data_points, yy = data_points.flatten(), yy.flatten()
            data_points = np.asarray([[x, y] for y in yy for x in data_points])
            evaluations = self.evaluate(
                gaussian_process, objective_function, data_points
            )
            return data_points[np.argmax(evaluations)]
        else:
            initial_point = np.random.uniform(*zip(*boundaries))
            result = optimize.fmin_l_bfgs_b(
                func=evaluate_gaussian_process,
                x0=initial_point,
                bounds=boundaries,
                approx_grad=True,
            )
            min_point = result[0]
            return min_point

    def __call__(
        self,
        gaussian_process: GaussianProcess,
        objective_function: ObjectiveFunction,
        data_points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self.evaluate(gaussian_process, objective_function, data_points)

    def evaluate(
        self,
        gaussian_process: GaussianProcess,
        objective_function: ObjectiveFunction,
        data_points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        If some values in data_points are supposed to be integers, we convert
        them to integers here before calling
        the method _evaluate

        :param gaussian_process:
        :param objective_function:
        :param data_points: numpy array of dimension n x m where n is the
        number of elements to evaluate
        and m is the number of variables used to calculate the objective
        function
        :return: estimates the value of the acquisition function for each
        element depending on:
        - the predictions of the gaussian process
        """
        dataset_bounds = objective_function.dataset_bounds
        data_points = data_points.reshape((-1, len(dataset_bounds)))
        data_points = objective_function.floor_integer_parameters(data_points)
        return self._evaluate(gaussian_process, data_points)

    @abc.abstractmethod
    def _evaluate(
        self, gaussian_process: GaussianProcess, data_points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number
        of elements to evaluate and m is the number of variables used to
        calculate the objective function
        :return: a numpy array of shape n x 1 (or a float) representing the
        estimation of the acquisition function at each point
        """

    def plot(
        self,
        gaussian_process: GaussianProcess,
        objective_function: ObjectiveFunction,
        last_evaluated_point: npt.NDArray[np.float64],
    ):
        number_dimensions = len(objective_function.dataset_bounds)

        if number_dimensions == 1:
            mesh_grid = objective_function.get_mesh_grid([100])
            x_ticks = mesh_grid[0]

            array_dataset = gaussian_process.array_dataset.flatten()
            evaluations = self.evaluate(gaussian_process, objective_function, x_ticks)

            if array_dataset.ndim > 1:
                array_dataset = array_dataset.flatten()
            if evaluations.ndim > 1:
                evaluations = evaluations.flatten()

            plt.plot(x_ticks, evaluations)
            plt.scatter(
                array_dataset,
                np.full_like(array_dataset, np.min(evaluations)),
                color="b",
                marker="x",
            )
            plt.axvline(last_evaluated_point, color="r")

            plt.title("Acquisition Function")
            plt.show()

        elif number_dimensions == 2:
            mesh_grid = objective_function.get_mesh_grid([100, 100])
            x_ticks, y_ticks = mesh_grid
            x_ticks, y_ticks = x_ticks.flatten(), y_ticks.flatten()
            data_points = np.asarray([[x, y] for y in y_ticks for x in x_ticks])
            evaluations = self.evaluate(
                gaussian_process, objective_function, data_points
            )
            evaluations = evaluations.reshape((x_ticks.size, y_ticks.size))

            contour_levels = [np.percentile(evaluations, k) for k in range(101)]
            contour_levels = sorted(list(set(contour_levels)))

            CS = plt.contourf(x_ticks, y_ticks, evaluations, levels=contour_levels)
            plt.scatter(
                gaussian_process.array_dataset[:, 0].flatten(),
                gaussian_process.array_dataset[:, 1].flatten(),
                color="b",
                marker="+",
            )
            plt.scatter(
                last_evaluated_point[0], last_evaluated_point[1], color="r", marker="x"
            )

            plt.colorbar(CS)
            plt.title("Acquisition Function")
            plt.show()
