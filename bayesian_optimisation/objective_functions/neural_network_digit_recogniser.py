import numpy as np
import numpy.typing as npt
import tensorflow
import tensorflow.keras.backend
from gp.objective_function import ObjectiveFunction
from gp.parameter_category import TypeVariable
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class NeuralNetworkDigitRecogniser(ObjectiveFunction):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Scale the data to between 0 and 1.
    X_train: npt.NDArray[np.float64] = X_train / 255
    X_test: npt.NDArray[np.float64] = X_test / 255

    # Flatten arrays from (28, 28) to (784,).
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # Convert the labels to categorical (one-hot encoding).
    y_train: npt.NDArray[np.float64] = utils.to_categorical(y_train, 10)
    y_test: npt.NDArray[np.float64] = utils.to_categorical(y_test, 10)

    # Establish the input shape for our networks.
    input_shape: tuple[int, ...] = X_train[0].shape

    def evaluate_without_noise(
        self, data_points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the objective function without noise.

        Parameters:
            data_points (npt.NDArray[np.float64]): Array of hyperparameter configurations.
                Each configuration is expected to have six values:
                [log_learning_rate, num_dense_layers, num_input_nodes,
                 num_dense_nodes, batch_size, log_adam_decay].

        Returns:
            npt.NDArray[np.float64]: Array of evaluation scores (negative validation accuracy).
        """
        evaluations: list[float] = []
        for data_point in data_points:
            (
                log_learning_rate,
                num_dense_layers,
                num_input_nodes,
                num_dense_nodes,
                batch_size,
                log_adam_decay,
            ) = data_point

            learning_rate = float(np.exp(log_learning_rate))
            num_dense_layers = int(num_dense_layers)
            num_input_nodes = int(num_input_nodes)
            num_dense_nodes = int(num_dense_nodes)
            batch_size = int(batch_size)
            adam_decay = float(np.exp(log_adam_decay))

            activation_function = "relu"

            evaluations.append(
                self._evaluate_without_noise(
                    learning_rate,
                    num_dense_layers,
                    num_input_nodes,
                    num_dense_nodes,
                    activation_function,
                    batch_size,
                    adam_decay,
                )
            )

        return np.asarray(evaluations).reshape((-1, 1))

    def _evaluate_without_noise(
        self,
        learning_rate: float,
        num_dense_layers: int,
        num_input_nodes: int,
        num_dense_nodes: int,
        activation: str,
        batch_size: int,
        adam_decay: float,
    ) -> float:
        """
        Evaluate a single hyperparameter configuration without noise.

        Parameters:
            learning_rate (float): The initial learning rate.
            num_dense_layers (int): Number of dense layers.
            num_input_nodes (int): Number of nodes in the input layer.
            num_dense_nodes (int): Number of nodes in each dense layer.
            activation (str): Activation function to use.
            batch_size (int): Batch size for training.
            adam_decay (float): Decay rate for the learning rate schedule.

        Returns:
            float: Negative validation accuracy (to be minimized).
        """
        model = self.create_model(
            learning_rate,
            num_dense_layers,
            num_input_nodes,
            num_dense_nodes,
            activation,
            adam_decay,
        )
        # Train the model; 'blackbox' represents the training history.
        blackbox = model.fit(
            x=self.X_train,
            y=self.y_train,
            epochs=3,
            batch_size=batch_size,
            validation_split=0.15,
            verbose=0,
        )
        # Retrieve the validation accuracy from the last epoch.
        accuracy = blackbox.history["val_accuracy"][-1]

        # Delete the Keras model to free memory.
        del model

        # Clear the Keras session to avoid graph accumulation.
        tensorflow.keras.backend.clear_session()

        # The objective is to minimize, so return the negative accuracy.
        return -accuracy

    @property
    def dataset_bounds(self) -> tuple[tuple[tuple[float, float], TypeVariable], ...]:
        """
        Defines the bounds and types of variables for the objective function.

        For example, if dataset_bounds is equal to:
            (
                ((1, 2), TypeVariable.REAL),
                ((5, 10), TypeVariable.INTEGER),
            )
        it means the objective function depends on two variables:
            - the first one is a real number between 1 and 2,
            - the second one is an integer between 5 (inclusive) and 10 (exclusive).

        Returns:
            tuple[tuple[tuple[float, float], TypeVariable], ...]: Variable bounds and types.
        """
        return (
            ((np.log(1e-4), np.log(1e-2)), TypeVariable.REAL),  # log Learning rate
            ((1, 5 + 1), TypeVariable.INTEGER),  # Number of dense layers
            ((1, 512 + 1), TypeVariable.INTEGER),  # Number of input nodes
            ((1, 28 + 1), TypeVariable.INTEGER),  # Number of dense nodes
            ((1, 128 + 1), TypeVariable.INTEGER),  # Batch size
            ((np.log(1e-6), np.log(1e-2)), TypeVariable.REAL),  # log adam decay
        )

    @classmethod
    def create_model(
        cls,
        learning_rate: float,
        num_dense_layers: int,
        num_input_nodes: int,
        num_dense_nodes: int,
        activation: str,
        adam_decay: float,
    ) -> Sequential:
        """
        Create a Sequential model with the given hyperparameters.

        Parameters:
            learning_rate (float): Initial learning rate.
            num_dense_layers (int): Number of dense layers.
            num_input_nodes (int): Number of nodes in the input layer.
            num_dense_nodes (int): Number of nodes in each dense layer.
            activation (str): Activation function.
            adam_decay (float): Decay rate for the learning rate schedule.

        Returns:
            Sequential: Compiled Keras model.
        """
        if num_input_nodes <= 0:
            raise ValueError("`num_input_nodes` most be strictly positive.")
        if num_dense_nodes <= 0:
            raise ValueError("`num_dense_nodes` most be strictly positive.")
        model = Sequential()
        # Instead of passing input_shape to a Dense layer, add an explicit Input layer.
        model.add(Input(shape=cls.input_shape))
        # Add the first Dense layer.
        model.add(Dense(num_input_nodes, activation=activation))
        # Add the specified number of dense layers.
        for i in range(num_dense_layers):
            name: str = f"layer_dense_{i + 1}"
            model.add(Dense(num_dense_nodes, activation=activation, name=name))
        # Add the final classification layer.
        model.add(Dense(10, activation="softmax"))

        # Create a learning rate schedule instead of using the deprecated 'decay' parameter.
        lr_schedule: tensorflow.keras.optimizers.schedules.ExponentialDecay = (
            tensorflow.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,  # Adjust based on your dataset and batch size.
                decay_rate=adam_decay,
                staircase=True,
            )
        )
        adam: Adam = Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
