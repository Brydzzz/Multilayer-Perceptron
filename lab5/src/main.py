import sklearn.datasets
from xor import (
    sigmoid,
    sigmoid_derv,
    softmax,
    softmax_derv,
    avg_square_loss,
    avg_sqr_derv,
    relu,
    relu_der,
)
import numpy as np
import pandas as pd
from mlp import MLP
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def one_hot(number: int):
    encode = np.zeros(shape=(10, 1))
    encode[number] = 1
    return encode


if __name__ == "__main__":
    # mnsint test data
    digits = load_digits(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.3, random_state=42
    )

    X_train["target"] = y_train
    # X_test["target"] = y_test
    mlp = MLP(
        layers_sizes=[64, 100, 70, 40, 10],
        loss_func=avg_square_loss,
        activation_func=sigmoid,
        loss_derv=avg_sqr_derv,
        activation_derv=sigmoid_derv,
        output_func=softmax,
        output_derv=softmax_derv,
        target_fit=one_hot,
    )

    mlp.train(
        training_data=X_train,
        epochs=100,
        mini_batch_size=10,
        learning_rate=0.5,
        class_column="target",
    )

    results = mlp.predict(X_test)

    def softmax_to_digits(array):
        exp_values = np.exp(array - np.max(array))
        probabilities = exp_values / np.sum(exp_values)

        return np.argmax(probabilities)

    classes = [softmax_to_digits(result) for result in results]
    # accuracy = np.mean(np.abs(results - xor_results) < 0.001)
    accuracy = sklearn.metrics.accuracy_score(y_test, classes)
    print(f"Results: {classes}")
    print(f"Should be: {y_test}")
    print(f"Accuracy: {accuracy}")
