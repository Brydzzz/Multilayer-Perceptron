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


if __name__ == "__main__":
    # xor train data
    xor_inputs = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
    xor_results = np.array([[0], [1], [1], [0]])
    xor_data = pd.DataFrame(
        data=np.hstack((xor_inputs, xor_results)),
        columns=["input_1", "input_2", "target"],
    )
    validation_data = xor_data.copy()
    mlp = MLP(
        layers_sizes=[2, 5, 4, 3, 1],
        loss_func=avg_square_loss,
        activation_func=relu,
        loss_derv=avg_sqr_derv,
        activation_derv=relu_der,
    )

    mlp.train(
        training_data=xor_data,
        epochs=5,
        mini_batch_size=2,
        learning_rate=0.01,
        class_column="target",
    )

    mlp.predict(validation_data)
