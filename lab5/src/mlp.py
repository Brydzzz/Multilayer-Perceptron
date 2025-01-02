import numpy as np
from typing import Callable, Tuple


class MLP:
    def __init__(
        self,
        layers_sizes: list[int],
        loss_func: Callable,
        activation_func: Callable[[np.ndarray], np.ndarray],
        loss_derv: Callable,
    ):
        self.weights = self.__init_weights(layers_sizes)
        self.biases = self.__init_biases(layers_sizes)
        self.loss = loss_func
        self.activation = activation_func
        self.loss_derv = loss_derv

    def __init_weights(self, lsizes):
        return [
            np.random.uniform(low=-1, high=1, size=(nout, nin))
            for nout, nin in zip(lsizes[1:], lsizes[:-1])
        ]

    def __init_biases(self, lsizes):
        return [
            np.random.uniform(low=-1, high=1, size=(n)) for n in lsizes[1:]
        ]

    def feed_forward(self, a: np.ndarray) -> np.ndarray:
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return a

    def get_activations_and_zs(
        self, a: np.ndarray
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        activs = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation(z)
            activs.append(a)
        return activs, zs

    def backward(
        self, x: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def train(
        self,
        training_data: np.ndarray,
        epochs: int,
        mini_batch_size: int,
        learning_rate: float,
    ) -> None:
        pass

    def predict(self, X: np.ndarray) -> list[int]:
        """
        A method that returns predicted number
        for each sample in MNIST dataset.

        Args:
            X (np.ndarray): The MNIST dataset.

        Returns:
            list[int]: Predicted number for each sample in X.
        """
        pass
