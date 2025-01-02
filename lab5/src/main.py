import numpy as np
from mlp import MLP
import math

# xor train data
xor_inputs = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
xor_results = np.array([[0], [1], [1], [0]])


def relu(sum: np.ndarray):
    return np.maximum(0, sum)


def relu_der(sum):
    return np.where(sum > 0, 1, 0)


def avg_square_loss(sum, target):
    return 0.5 * ((sum - target) ** 2)


def avg_sqr_derv(sum, target):
    return sum - target


mlp = MLP(
    layers_sizes=[2, 10, 1],
    loss_func=avg_square_loss,
    activation_func=relu,
    loss_derv=avg_sqr_derv,
    activation_derv=relu_der,
)
mlp.backward(xor_inputs[2].reshape(xor_inputs[2].shape[0], 1), xor_results[2])
if __name__ == "__main__":
    pass
