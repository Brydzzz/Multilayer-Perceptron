import numpy as np
from typing import Callable


class Neuron:
    def __init__(self, nin):
        self.weights = [np.random.uniform(-1, 1) for _ in range(nin)]
        self.biases = [np.random.uniform(-1, 1) for _ in range(nin)]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]


class MLP:
    def __init__(
        self,
        layers_num: int,
        neurons_num: np.ndarray,
        loss_func: Callable,
        activation_func: Callable,
    ):
        if len(neurons_num) != layers_num:
            raise ValueError(
                "Neuron numbers length do not match layers number"
            )
        self.layers = self.__init_layers(layers_num, neurons_num)
        self.loss = loss_func
        self.activation = activation_func

    def __init_layers(self, lnum, nnum):
        layers = []
        for i in range(lnum):
            if i == 0:
                layers.append(Layer(0, nnum[i + 1]))
            elif i == lnum - 1:
                layers.append(Layer(nnum[i], 0))
            else:
                layers.append(Layer(nnum[i], nnum[i + 1]))  # Hidden Layers
        return layers

    def forward(self):
        pass

    def backward(self):
        pass

    def train(self, epoch):
        pass

    def predict(self):
        pass
