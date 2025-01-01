import numpy as np
from typing import Callable, Tuple



# TODO: get rid of Layer class
class Layer:
    def __init__(self, nout, nin, biases):
        # self.weights = [
        #     [np.random.uniform(-1, 1) for _ in range(nin)] for _ in range(nout)
        # ]
        # self.biases = [
        #     [np.random.uniform(-1, 1) for _ in range(nin)] for _ in range(nout)
        # ]
        """
        self.weights -> input weights for each neuron in the layer
        self.biases -> input biases for each neuron in the layer
        """
        self.weights = np.random.uniform(-1, 1, size=(nout, nin))
        self.biases = (
            biases if biases else np.random.uniform(-1, 1, size=(nout))
        )

    def calculate_z(self, activations):
        return np.dot(self.weights, activations) + self.biases


class MLP:
    def __init__(
        self,
        layers_sizes: list[int],
        loss_func: Callable,
        activation_func: Callable[[np.ndarray], np.ndarray],
        loss_derv: Callable[[np.n]],
    ):
        self.layers = self.__init_layers(layers_sizes)
        self.loss = loss_func
        self.activation = activation_func
        self.loss_derv = loss_derv

    def __init_layers(self, lsizes):
        layers = []
        for i in range(lsizes):
            if i == 0:
                layers.append(Layer(0, 0, biases=np.zeros(lsizes[i])))

            else:
                layers.append(Layer(lsizes[i], lsizes[i - 1]))  # Hidden Layers
        return layers

    def feed_forward(
        self, a: np.ndarray
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        activs = [a]
        zs = []
        for layer in self.layers:
            z = layer.calculate_z(a)
            zs.append(z)
            a = self.activation(z)
            activs.append(a)

        return activs, zs

    def backward(self, targets: np.ndarray):
        pass

    def train(self, epoch):
        pass

    def predict(self):
        pass
