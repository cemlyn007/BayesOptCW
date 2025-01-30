import copy
from typing import Generator

import numpy as np
from gp.gaussian_process import GaussianProcess
from gp.kernels.kernel import Kernel
from gp.objective_function import ObjectiveFunction

from .acquisition_functions.abstract_acquisition_function import AcquisitionFunction


class BayesianOptimisation:
    def __init__(
        self,
        kernel: Kernel,
        objective_function: ObjectiveFunction,
        acquisition_function: AcquisitionFunction,
        verbose=False,
    ):
        """
        :param kernel: Kernel object used by the gaussian process to perform
        a regression.
        :param objective_function: ObjectiveFunction object which we will try to
        minimise
        :param acquisition_function: AcquisitionFunction object
        """
        self._initial_kernel = copy.deepcopy(kernel)
        self._gaussian_process = GaussianProcess(kernel)
        self._objective_function = objective_function
        self._acquisition_function = acquisition_function
        self.verbose = verbose

    def _initialise_gaussian_process(
        self,
        dataset: npt.NDArray[np.float64],
        objective_function_values: npt.NDArray[np.float64],
    ) -> None:
        """
        Initialise the gaussian process with its initial dataset
        :param dataset: array representing all the data points
        used to calculate the posterior mean and variance of the GP.
        Its dimension is n x l, there are:
        - n elements in the dataset. Each row corresponds to a data point x_i
        (with 1<=i<=n), at which the objective function can be evaluated
        - each one of them is of dimension l (representing the number of
        variables required by the objective function)
        :param objective_function_values: array of the
        evaluations for all the elements in array_dataset.
        Its shape is hence n x 1 (it's a column vector)
        """

        self._gaussian_process.initialise_dataset(dataset, objective_function_values)

    def run(
        self,
        number_steps: int,
        initial_dataset: npt.NDArray[np.float64],
        initial_objective_function_values: npt.NDArray[np.float64],
    ) -> Generator:
        """
        Generator that performs a bayesian optimisation

        This method is a generator: at every step, it yields a tuple containing
        3 elements:
        - the current up-to-date gaussian process
        - the acquisition function
        - the last computed argmax of the acquisition function.

        Hence, in order to use this method, you need to put it in a for loop,
            for gp, af, arg_max in bo.run(): # Here, bo is a
            BayesianOptimisation object
                # some code here


        :param number_steps: number of steps to execute in the Bayesian
        Optimisation procedure.

        :param initial_dataset: array_initial_dataset: array representing all
        the data points used to calculate the posterior mean and variance of
        the GP.
        Its dimension is n x l, there are:
        - n elements in the dataset. Each row corresponds to a data point x_i
        (with 1<=i<=n), at which the objective function can be evaluated
        - each one of them is of dimension l (representing the number of
        variables required by the objective function)

        :param initial_objective_function_values: array of the evaluations for
        all the elements in array_dataset. Its shape is hence n x 1
        (it's a column vector)
        """
        if self.verbose:
            print(
                f"Step {0}/{number_steps} "
                f"- Initialise Gaussian Process for Provided Dataset"
            )
        self._initialise_gaussian_process(
            initial_dataset, initial_objective_function_values
        )
        arg_max_of_acquisition_function = self.compute_arg_max_of_acquisition_function()

        for index_step in range(1, number_steps + 1):
            if self.verbose:
                print(
                    f"Step {index_step}/{number_steps} "
                    f"- Evaluating Objective Function at position: "
                    f"{arg_max_of_acquisition_function.tolist()}"
                )
            arg_max_of_acquisition_function = self._step(
                arg_max_of_acquisition_function
            )
            yield (
                self._gaussian_process,
                self._acquisition_function,
                arg_max_of_acquisition_function,
            )

    def _step(
        self, arg_max_acquisition_function: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param arg_max_acquisition_function: the previously computed argmax of
        the acquisition function
        :return: the next computed arg_max of the acquisition function after
        having updated the Gaussian Process
        """
        y = self._objective_function.evaluate(arg_max_acquisition_function).item()
        self._gaussian_process.add_data_point(arg_max_acquisition_function, y)
        self._gaussian_process.optimise_parameters(disp=False)
        return self.compute_arg_max_of_acquisition_function()

    def get_best_data_point_index(self) -> npt.NDArray[np.float64]:
        return np.argmin(self._gaussian_process.array_objective_function_values)

    def get_best_data_point(self) -> npt.NDArray[np.float64]:
        best_index = self.get_best_data_point_index()
        return self._gaussian_process.array_dataset[best_index]

    def compute_arg_max_of_acquisition_function(self) -> npt.NDArray[np.float64]:
        return self._acquisition_function.compute_arg_max(
            gaussian_process=self._gaussian_process,
            objective_function=self._objective_function,
        )

    def reinitialise_kernel(self) -> None:
        self._gaussian_process.set_kernel_parameters(
            self._initial_kernel.log_amplitude,
            self._initial_kernel.log_length_scale,
            self._initial_kernel.log_noise_scale,
        )
