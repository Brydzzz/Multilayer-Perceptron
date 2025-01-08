import numpy as np


def relu(sum: np.ndarray):
    return np.maximum(0, sum)


def relu_derv(sum):
    return np.where(sum > 0, 1, 0)


def avg_square_loss(sum, target):
    return 0.5 * ((sum - target) ** 2)


def avg_sqr_derv(sum, target):
    return sum - target


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def softmax(x):
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def softmax_derv(x):
    s = softmax(x)
    return s * (1 - s)


def none(x):
    return x


def none_derv(x):
    return np.zeros_like(x)


def six_init(nout, nin):
    return np.random.uniform(
        low=-np.sqrt(6 / nin),
        high=np.sqrt(6 / nin),
        size=(nout, nin),
    )


def basic_bias(n):
    return np.random.uniform(low=-1, high=1, size=(n, 1))
