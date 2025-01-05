import numpy as np
import pandas as pd
from typing import Callable, Tuple


class MLP:
    def __init__(
        self,
        layers_sizes: list[int],
        loss_func: Callable,
        activation_func: Callable[[np.ndarray], np.ndarray],
        loss_derv: Callable,
        activation_derv: Callable,
    ):
        self.weights = self.__init_weights(layers_sizes)
        self.biases = self.__init_biases(layers_sizes)
        self.loss = loss_func
        self.activation = activation_func
        self.loss_derv = loss_derv
        self.activation_derv = activation_derv

    def __init_weights(self, lsizes):
        return [
            np.random.uniform(low=-1, high=1, size=(nout, nin))
            for nout, nin in zip(lsizes[1:], lsizes[:-1])
        ]

    def __init_biases(self, lsizes):
        return [
            np.random.uniform(low=-1, high=1, size=(n, 1)) for n in lsizes[1:]
        ]

    def feed_forward(self, a: np.ndarray) -> np.ndarray:
        a = a.reshape(a.shape[0], 1)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return a

    def get_activations_and_zs(
        self, a: np.ndarray
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        a = a.reshape(a.shape[0], 1)
        activs = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation(z)
            activs.append(a)
        return activs, zs

    def backward(
        self, a: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        weight_dervs = [np.zeros(w.shape) for w in self.weights]
        bias_dervs = [np.zeros(b.shape) for b in self.biases]
        activs, zs = self.get_activations_and_zs(a)
        loss_derv = self.loss_derv(activs[-1], targets) * self.activation_derv(
            zs[-1]
        )
        bias_dervs[-1] = loss_derv
        weight_dervs[-1] = np.dot(loss_derv, activs[-2].transpose())
        for i in range(len(self.weights) - 2, -1, -1):
            loss_derv = np.dot(
                self.weights[i + 1].transpose(), loss_derv
            ) * self.activation_derv(zs[i])
            bias_dervs[i] = loss_derv
            weight_dervs[i] = np.dot(loss_derv, activs[i].transpose())
        return weight_dervs, bias_dervs

    def train(
        self,
        training_data: pd.DataFrame,
        epochs: int,
        mini_batch_size: int,
        learning_rate: float,
        class_column: str,
    ) -> None:
        for epoch in range(epochs):
            mini_batches = self.initialize_mini_batches(
                training_data, mini_batch_size
            )
            for batch in mini_batches:
                self.process_batch(batch, class_column, learning_rate)
            print(f"Epoch {epoch + 1} completed.")

    def process_batch(self, batch, class_column, learning_rate):
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        classes = batch[class_column].to_numpy()
        inputs = batch.drop(columns=[class_column]).to_numpy()
        for i in range(len(classes)):
            weights_dervs, biases_dervs = self.backward(inputs[i], classes[i])
            weight_gradient = [
                wg + wd for wg, wd in zip(weight_gradient, weights_dervs)
            ]
            bias_gradient = [
                bg + bd for bg, bd in zip(bias_gradient, biases_dervs)
            ]
        self.weights = [
            weight - learning_rate * w_grad / len(inputs)
            for weight, w_grad in zip(self.weights, weight_gradient)
        ]
        self.biases = [
            bias - learning_rate * b_grad / len(inputs)
            for bias, b_grad in zip(self.biases, bias_gradient)
        ]

    def initialize_mini_batches(
        self, t_data: pd.DataFrame, mini_batch_size: int
    ):
        mini_batches = []
        while not t_data.empty:
            mini_batch = t_data.sample(n=min(mini_batch_size, len(t_data)))
            mini_batches.append(mini_batch)
            t_data = t_data.drop(mini_batch.index)
        return mini_batches

    def predict(self, X: pd.DataFrame) -> list[int]:
        """
        A method that returns predicted class
        from the dataset given.

        Args:
            X (pd.DataFrame): The dataset.

        Returns:
            list[int]: Predicted class
        """
        predictions = []
        for x in X:
            prediction = self.feed_forward(x)
            predictions.append(prediction)
        return predictions
