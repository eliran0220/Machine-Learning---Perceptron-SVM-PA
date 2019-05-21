import numpy as np

"""
The Perceptron class, it takes parameters and initializes them, if they are not given.
:param epochs: the number of iterations to be made
:param eta: the learning rate of the algorithm
:param training_set: the given training set to be trained
:param label_set: the given label set
:param seed: the given seed for the norm function
"""


class Perceptron:
    def __init__(self, ephocs=100, eta=0.01, training_set=None, label_set=None, seed=None):
        self._ephocs = ephocs
        self._eta = eta
        self._training_set = training_set
        self._label_set = label_set
        self._seed = seed

    """
    The training function
    :param self: self values
    :param training_set: the given training set to be trained
    :param label_set: the given label set
    :param weight: the given weights
    """

    def training(self, training_set=None, label_set=None, weight=None):
        w = weight.copy()
        if training_set is None:
            training = self._training_set
            labels = self._label_set
        else:
            training = training_set
            labels = label_set
        for e in range(self._ephocs):
            # using the seed for proper permutation
            np.random.seed(self._seed[e])
            permutation = np.random.permutation(training.shape[0])
            training = training[permutation]
            labels = labels[permutation]
            # here we update the values according to the condition
            for i in range(len(training)):
                x = training[i]
                y = labels[i]
                y_hat = np.argmax(np.dot(w, x))
                if y[0] != y_hat:
                    w[int(y[0]), :] = w[int(y[0]), :] + self._eta * x
                    w[y_hat, :] = w[y_hat, :] - self._eta * x
        return w