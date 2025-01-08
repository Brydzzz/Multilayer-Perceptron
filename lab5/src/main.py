import sklearn.datasets
from lab5.src.funcs import (
    one_hot,
    sigmoid,
    sigmoid_derv,
    softmax,
    softmax_derv,
    avg_square_loss,
    avg_sqr_derv,
    relu,
    relu_derv,
    six_init,
    basic_bias,
    softmax_to_digits,
)
import numpy as np
from mlp import MLP
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def scale_row(X):
    new_row = np.zeros(X.shape)
    for i, x in enumerate(X):
        new_row[i] = (x - np.min(X)) / (np.max(X) - np.min(X))
    return new_row


def rescale_inputs(X):
    for idx, row in X.iterrows():
        row = scale_row(row.to_numpy())
        X.loc[idx] = row
    return X


if __name__ == "__main__":
    # mnsint test data
    digits = load_digits(as_frame=True)
    train_set, test_set, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.6, random_state=42
    )
    train_set = rescale_inputs(train_set)
    test_set = rescale_inputs(test_set)
    train_set["target"] = y_train
    test_set, valid_set, y_test, y_val = train_test_split(
        test_set, y_test, test_size=0.5, random_state=42
    )
    mlp = MLP(
        layers_sizes=[64, 128, 64, 32, 16, 8, 10],
        loss_func=avg_square_loss,
        activation_func=relu,
        loss_derv=avg_sqr_derv,
        activation_derv=relu_derv,
        output_func=softmax,
        output_derv=softmax_derv,
        target_fit=one_hot,
        weight_init=six_init,
        bias_init=basic_bias,
    )

    mlp.train(
        training_data=train_set,
        epochs=100,
        mini_batch_size=10,
        learning_rate=0.55,
        class_column="target",
    )

    results = mlp.predict(test_set)

    classes = [softmax_to_digits(result) for result in results]
    accuracy = sklearn.metrics.accuracy_score(y_test, classes)
    print(f"Accuracy: {accuracy}")
