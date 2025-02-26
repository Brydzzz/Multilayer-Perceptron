# Multilayer Perceptron From Scratch In Python

Made by Brydzzz and Elbergg for Introduction to Artificial Intelligence class

## Implementation

The Perceptron is in `mlp.py`. It is a general implementation which can use:

- different activation functions (for both hidden and output layers)
- custom weight and bias initializations
- different loss functions
  This all makes the implementation suited for many different tasks.
  It uses Stochastic Gradient Descent by splitting the data into mini batches.
  Libraries used:
- `numpy`
- `pandas`

## Showcase

The showcase is in `main.py`.
We train the model on hand drawng digits from the MNIST dataset, and then check it's performance on the testing set.
Functions used:
- average square loss function
- ReLu hidden layer acitvation
- Softmax output activation
We use one hot encoding for classifying the results.
