import matplotlib.pyplot as plt
import numpy as np

from acquisition_functions.lower_confidence_bound import LowerConfidenceBound
from bayesian_optimisation import BayesianOptimisation
from gaussian_process import GaussianProcess
from kernels.gaussian_kernel import GaussianKernel
from objective_functions.six_hump_camel import SixHumpCamelObjectiveFunction
from objective_functions.univariate_objective_function import UnivariateObjectiveFunction
from kernels.matern_kernel import MaternKernel
from objective_functions.six_hump_camel import SixHumpCamelObjectiveFunction
from acquisition_functions.expected_improvement import ExpectedImprovement

# Some examples of functions we are gonna work on and their noisy versions


plt.title("Univariate Objective Function")
UnivariateObjectiveFunction().plot([100])

plt.title("Noisy Univariate Objective Function")
UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5).plot([100])

plt.title("Six Hump Camel Objective Function")
SixHumpCamelObjectiveFunction().plot([100, 100])

plt.title("Noisy Six Hump Camel Objective Function")
SixHumpCamelObjectiveFunction(additional_gaussian_noise_std=0.3).plot([100, 100])

objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_gaussian = GaussianKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_gaussian)

gaussian_process.plot(objective_function, show=False)
plt.title("Gaussian Process Regression - no dataset")
plt.show()

boundaries, = objective_function.boundaries
x = np.linspace(*boundaries, 50).reshape((-1, 1))
y = objective_function.evaluate(x).reshape((-1, 1))
gaussian_process.initialise_dataset(x, y)

gaussian_process.plot(objective_function, show=False)
plt.title("Gaussian Process Regression - with dataset")
plt.show()

## Task 3: Function Sampling (10 marks)
objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_gaussian = GaussianKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_gaussian)

gaussian_process.plot_with_samples(5, objective_function)  # 5
plt.show()

boundaries, = objective_function.boundaries
x = np.linspace(*boundaries, 50).reshape((-1, 1))
y = objective_function.evaluate(x).reshape((-1, 1))
gaussian_process.initialise_dataset(x, y)

gaussian_process.plot_with_samples(5, objective_function)  # 5
plt.show()

# # ## Task 4: Matern Kernel (10 marks)

objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_matern = MaternKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_matern)

gaussian_process.plot_with_samples(5, objective_function)
plt.show()

boundaries, = objective_function.boundaries
x = np.linspace(*boundaries, 50).reshape((-1, 1))
y = objective_function.evaluate(x).reshape((-1, 1))
gaussian_process.initialise_dataset(x, y)

gaussian_process.plot_with_samples(5, objective_function)
plt.show()

# # ## Task 5: Marginal Likelihood (20 marks)

objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_gaussian = GaussianKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_gaussian)

boundaries, = objective_function.boundaries
x = np.linspace(*boundaries, 50).reshape((-1, 1))
y = objective_function.evaluate(x).reshape((-1, 1))
gaussian_process.initialise_dataset(x, y)

gaussian_process.optimise_parameters(disp=True)
gaussian_process.plot(objective_function, show=False)
plt.show()

objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_matern = MaternKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_matern)

boundaries, = objective_function.boundaries
x = np.linspace(*boundaries, 50).reshape((-1, 1))
y = objective_function.evaluate(x).reshape((-1, 1))
gaussian_process.initialise_dataset(x, y)

gaussian_process.optimise_parameters(disp=True)
gaussian_process.plot(objective_function, show=False)
plt.show()

# # ## Task 6: Implement metrics, to measure performance on test set (10 marks)

objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_gaussian = GaussianKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_gaussian)

boundaries, = objective_function.boundaries
x_train = np.linspace(*boundaries, 50).reshape((-1, 1))
y_train = objective_function.evaluate(x_train).reshape((-1, 1))

x_test = np.linspace(*boundaries, 150).reshape((-1, 1))
y_test = objective_function.evaluate(x_test).reshape((-1, 1))

gaussian_process.initialise_dataset(x_train, y_train)
print(f"LPD Gaussian Process without marginal likelihood optimisation: "
      f"{gaussian_process.get_log_predictive_density(x_test, y_test)}")

gaussian_process.plot(objective_function)

print('-' * 50)
gaussian_process.optimise_parameters(disp=False)

print(f"LPD optimised Gaussian Process: "
      f"{gaussian_process.get_log_predictive_density(x_test, y_test)}")
gaussian_process.plot(objective_function)

# # ## Task 8: Bayesian Optimisation (10 marks)

# # ---------------
# # Bayesian Optimisation in 1 dimension
# # ---------------

kernel = GaussianKernel(-1., -1., -1.)

objective_function = UnivariateObjectiveFunction()
# objective_function = objective_functions.six_hump_camel.SixHumpCamelObjectiveFunction()

# acquisition_function = LowerConfidenceBound(2.)
acquisition_function = ExpectedImprovement()

bayesian_optimisation = BayesianOptimisation(
    kernel,
    objective_function,
    acquisition_function
)

number_initial_elements = 1
print(f"Initialising Dataset with {number_initial_elements} Initial Elements")
dataset = objective_function.get_random_initial_dataset(number_initial_elements)
evaluations = objective_function.evaluate(dataset)

number_steps_bayesian_optimisation = 25
print(f"Launching Bayesian Optimisation with {number_steps_bayesian_optimisation} Steps")

bo_generator = bayesian_optimisation.run(
    number_steps=number_steps_bayesian_optimisation,
    array_initial_dataset=dataset,
    array_initial_objective_function_values=evaluations
)

for gp, aq, arg_max_aq in bo_generator:
    print(objective_function.boundaries)
    boundaries_x, = objective_function.boundaries
    plt.xlim(boundaries_x)
    plt.ylim(-3., 3.)

    gp.plot(objective_function)

    plt.xlim(boundaries_x)
    aq.plot(gp, objective_function, last_evaluated_point=arg_max_aq)

print(f"Best argmin found for the objective function: {bayesian_optimisation.get_best_data_point()}")

# # ---------------
# # Bayesian Optimisation in 2 dimensions
# # ---------------
#

kernel = GaussianKernel(0.5, np.log(1.), 1 * np.log(1.))

# objective_function = objective_functions.univariate_objective_function.UnivariateObjectiveFunction()
objective_function = SixHumpCamelObjectiveFunction()

# acquisition_function = LowerConfidenceBound(2.)
acquisition_function = ExpectedImprovement()

bayesian_optimisation = BayesianOptimisation(
    kernel,
    objective_function,
    acquisition_function
)

number_initial_elements = 4
print(f"Initialising Dataset with {number_initial_elements} Initial Elements")
dataset = objective_function.get_random_initial_dataset(number_initial_elements)
evaluations = objective_function.evaluate(dataset)

number_steps_bayesian_optimisation = 25
print(f"Launching Bayesian Optimisation with {number_steps_bayesian_optimisation} Steps")

bo_generator = bayesian_optimisation.run(
    number_steps=number_steps_bayesian_optimisation,
    array_initial_dataset=dataset,
    array_initial_objective_function_values=evaluations
)
show = True
for gp, aq, arg_max_aq in bo_generator:
    if show:
        boundaries_x, boundaries_y = objective_function.boundaries
        plt.xlim(boundaries_x)
        plt.ylim(boundaries_y)
        gp.plot(objective_function)
        plt.xlim(boundaries_x)
        aq.plot(gp, objective_function, last_evaluated_point=arg_max_aq)
if not show:
    boundaries_x, boundaries_y = objective_function.boundaries
    plt.xlim(boundaries_x)
    plt.ylim(boundaries_y)
    gp.plot(objective_function)
    plt.xlim(boundaries_x)
    aq.plot(gp, objective_function, last_evaluated_point=arg_max_aq)

print(f"Best argmin found for the objective function: {bayesian_optimisation.get_best_data_point()}")

# # ### Questions
# #
# # Question 1: What happens if there is not enough data provided at the Initialisation Step?

# ---------------
# Bayesian Optimisation in 2 dimensions Case with only 1 initial elements!
# ---------------

kernel = GaussianKernel(0.5, np.log(1.), 1 * np.log(1.))

objective_function = SixHumpCamelObjectiveFunction()

acquisition_function = ExpectedImprovement()

bayesian_optimisation = BayesianOptimisation(
    kernel,
    objective_function,
    acquisition_function
)

number_initial_elements = 1
print(f"Initialising Dataset with {number_initial_elements} Initial Elements")
dataset = objective_function.get_random_initial_dataset(number_initial_elements)
evaluations = objective_function.evaluate(dataset)

number_steps_bayesian_optimisation = 150
print(f"Launching Bayesian Optimisation with {number_steps_bayesian_optimisation} Steps")

bo_generator = bayesian_optimisation.run(
    number_steps=number_steps_bayesian_optimisation,
    array_initial_dataset=dataset,
    array_initial_objective_function_values=evaluations
)

show = True
for gp, aq, arg_max_aq in bo_generator:
    if show:
        boundaries_x, boundaries_y = objective_function.boundaries
        plt.xlim(boundaries_x)
        plt.ylim(boundaries_y)
        gp.plot(objective_function)
        plt.xlim(boundaries_x)
        aq.plot(gp, objective_function, last_evaluated_point=arg_max_aq)
if not show:
    boundaries_x, boundaries_y = objective_function.boundaries
    plt.xlim(boundaries_x)
    plt.ylim(boundaries_y)
    gp.plot(objective_function)
    plt.xlim(boundaries_x)
    aq.plot(gp, objective_function, last_evaluated_point=arg_max_aq)

print(f"Best argmin found for the objective function: {bayesian_optimisation.get_best_data_point()}")

print("#######################################################")
print("Neural Network Digit Recogniser")
from objective_functions.neural_network_digit_recogniser import NeuralNetworkDigitRecogniser
from acquisition_functions.expected_improvement import ExpectedImprovement

kernel = GaussianKernel(0.5, np.log(1.), 1 * np.log(1.))

objective_function = NeuralNetworkDigitRecogniser()

acquisition_function = ExpectedImprovement()

bayesian_optimisation = BayesianOptimisation(
    kernel,
    objective_function,
    acquisition_function
)

number_initial_elements = 4
print(f"Initialising Dataset with {number_initial_elements} Initial Elements")
dataset = objective_function.get_random_initial_dataset(number_initial_elements)
evaluations = objective_function.evaluate(dataset)

number_steps_bayesian_optimisation = 60
print(f"Launching Bayesian Optimisation with {number_steps_bayesian_optimisation} Steps")

bo_generator = bayesian_optimisation.run(
    number_steps=number_steps_bayesian_optimisation,
    array_initial_dataset=dataset,
    array_initial_objective_function_values=evaluations
)

for gp, aq, arg_max_aq in bo_generator:
    continue

print(f"Best argmin found for the objective function: {bayesian_optimisation.get_best_data_point()}")

print(objective_function.evaluate(np.array([-5.22402802, 1.19836661, 405.13074034, 23.56388671, 43.66787787,
                                            -5.35928838])))

# [ -7.85040982   1.63370621 480.08725509  15.73128475   7.05940531, -12.64395237]
