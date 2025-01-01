import numpy as np
from mlp import MLP
import math

# xor train data
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_results = np.array([0], [1], [1], [0])


def relu(sum):
    return max(0, sum)


def avg_square_loss(sum, target):
    return 1 / 2(**math.abs(sum - target))


mlp = MLP(
    layers_num=5,
    nuerons_num=[3, 3, 3, 3, 3],
    loss_func=avg_square_loss,
    activation_func=relu,
)
if __name__ == "__main__":
    pass
