import numpy as np
import pandas as pd
from mlp import MLP


def relu(sum: np.ndarray):
    return np.maximum(0, sum)


def relu_der(sum):
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


if __name__ == "__main__":
    # xor train data
    xor_inputs = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
    xor_results = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    xor_data = pd.DataFrame(
        {
            "input_1": xor_inputs[:, 0],
            "input_2": xor_inputs[:, 1],
            "target": [arr for arr in xor_results],
        }
    )
    validation_data = xor_data.copy().drop("target", axis=1)
    mlp = MLP(
        layers_sizes=[2, 20, 2],
        loss_func=avg_square_loss,
        activation_func=sigmoid,
        loss_derv=avg_sqr_derv,
        activation_derv=sigmoid_derv,
        output_func=softmax,
        output_derv=softmax_derv,
    )

    mlp.train(
        training_data=xor_data,
        epochs=10000,
        mini_batch_size=2,
        learning_rate=0.001,
        class_column="target",
    )

    results = mlp.predict(validation_data)

    def softmax_to_xor(arrays):
        probabilities = []
        for array in arrays:
            first_prob = array[0][0]
            second_prob = array[1][0]
            probabilities.append([first_prob, second_prob])
        classified = [
            [1, 0] if probs[0] > probs[1] else [0, 1]
            for probs in probabilities
        ]
        return np.array(classified)

    classes = softmax_to_xor(results)
    # accuracy = np.mean(np.abs(results - xor_results) < 0.001)
    print(f"Results: {classes}")
    print(f"Should be: {xor_results}")
    # print(f"Accuracy: {accuracy}")
