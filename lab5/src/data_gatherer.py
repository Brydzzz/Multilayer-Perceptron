import csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from lab5.src.funcs import (
    avg_sqr_derv,
    basic_bias,
    relu,
    relu_derv,
    six_init,
    softmax,
    softmax_derv,
    softmax_to_digits,
)
from lab5.src.main import one_hot
from lab5.src.mlp import MLP


class DataGatherer:
    def __init__(
        self,
        mini_batch_sizes: list[int],
        learning_rates: list[float],
        train_set: pd.Dataframe,
        valid_set: pd.Dataframe,
        y_valid: pd.Series,
        test_set: pd.Dataframe,
        y_test: pd.Series,
        class_column="target",
        layers_sizes=[64, 48, 32, 16, 8, 10],
        loss_func_derv=avg_sqr_derv,
        activation_func=relu,
        activation_derv=relu_derv,
        weight_init=six_init,
        bias_init=basic_bias,
        epochs=100,
    ):
        self.mini_batch_sizes = mini_batch_sizes
        self.learning_rates = learning_rates
        self.train_set = train_set
        self.valid_set = valid_set
        self.y_valid = y_valid
        self.y_test = y_test
        self.test_set = test_set
        self.class_column = class_column
        self.layers_sizes = layers_sizes
        self.loss_derv = loss_func_derv
        self.activation = activation_func
        self.activation_derv = activation_derv
        self.weight_init = weight_init
        self.bias_init = bias_init

    def _save_to_csv(self, data):
        with open("lab5/data/test.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Mini Batch Size",
                    "Learning Rate",
                    "Validation Set Accuracy",
                ]
            )
            writer.writerows(data)

    def _plot_results(self, mini_batch_results):
        results_array = np.array(mini_batch_results)
        unique_mb_sizes = np.unique(results_array[:, 0])

        plt.figure(figsize=(10, 6))

        for mb_size in unique_mb_sizes:
            mb_mask = results_array[:, 0] == mb_size
            mb_data = results_array[mb_mask]

            lr_values = mb_data[:, 1]
            accuracy_values = mb_data[:, 2]
            plt.plot(
                lr_values,
                accuracy_values,
                marker="o",
                label=f"Batch Size: {int(mb_size)}",
            )

        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")
        plt.title(
            "Validation Accuracy vs Learning Rate for Different Mini-batch Sizes"
        )
        plt.legend()
        plt.grid(True, which="minor", linestyle="--", alpha=0.4)
        plt.show()

    def _calculate_accuracy(self, predictions, on_test: bool = False):
        predicted_digits = [
            softmax_to_digits(prediction) for prediction in predictions
        ]
        y = self.y_test if on_test else self.y_valid
        accuracy = sklearn.metrics.accuracy_score(y, predicted_digits)
        return accuracy

    def _find_best_parameters(
        self, mini_batch_results
    ) -> tuple[int, float, float]:
        best_mb_size = None
        best_lr = None
        best_accuracy = float("-inf")

        for mb_size, lr, accuracy in mini_batch_results:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_mb_size = mb_size
                best_lr = lr

        return best_mb_size, best_lr, best_accuracy

    def _calculate_best_accuracy(self, best_parameters):
        best_mb_size, best_lr, _ = best_parameters
        mlp = MLP(
            self.layers_sizes,
            self.activation_func,
            self.loss_func_derv,
            self.activation_derv,
            softmax,
            softmax_derv,
            one_hot,
            self.weight_init,
            self.bias_init,
        )
        mlp.train(
            training_data=self.train_set,
            epochs=self.epochs,
            mini_batch_size=best_mb_size,
            learning_rate=best_lr,
            class_column=self.class_column,
        )
        predictions = mlp.predict(self.test_set)
        accuracy = self._calculate_accuracy(predictions, on_test=True)
        return accuracy

    def generate_data(
        self, save_to_csv: bool = True, plot_results: bool = True
    ):
        mini_batch_results = []
        for mb_size in self.mini_batch_sizes:
            # lr_accuracy = []
            for lr in self.learning_rates:
                mlp = MLP(
                    self.layers_sizes,
                    self.activation_func,
                    self.loss_func_derv,
                    self.activation_derv,
                    softmax,
                    softmax_derv,
                    one_hot,
                    self.weight_init,
                    self.bias_init,
                )
                mlp.train(
                    training_data=self.train_set,
                    epochs=self.epochs,
                    mini_batch_size=mb_size,
                    learning_rate=lr,
                    class_column=self.class_column,
                )
                predictions = mlp.predict(self.valid_set)
                # lr_accuracy.append(
                #     (lr, self._calculate_val_accuracy(predictions))
                # )
                mini_batch_results.append(
                    (mb_size, lr, self._calculate_accuracy(predictions))
                )

        best_parameters = self._find_best_parameters(mini_batch_results)
        best_accuracy = self._calculate_best_accuracy(best_parameters)
        print("Best configuration found:")
        print(f"Mini-batch size: {best_parameters[0]}")
        print(f"Learning rate: {best_parameters[1]}")
        print(f"Validation accuracy: {best_accuracy:.4f}")

        if save_to_csv:
            self._save_to_csv()
        if plot_results:
            self._plot_results()

        return mini_batch_results
