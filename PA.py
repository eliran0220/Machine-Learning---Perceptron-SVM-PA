import numpy as np

"""
The PA class, it takes parameters and initializes them, if they are not given.
:param self: self values
:param epochs: the number of iterations to be made
:param training_set: the given training set to be trained
:param label_set: the given label set
:param seed: the given seed for the norm function
"""


class PA:
    def __init__(self, ephocs=100, training_set=None, label_set=None, seed=None):
        self._ephocs = ephocs
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
            np.random.seed(self._seed[e])
            permutation = np.random.permutation(training.shape[0])
            training = training[permutation]
            labels = labels[permutation]
            for i in range(len(training)):
                x = training[i]
                y = labels[i]
                y_hat = np.argmax(np.dot(w, x))
                if y != y_hat:
                    loss = max(0, 1 - np.dot(w[int(y[0]), :], x) + np.dot(w[int(y_hat), :], x))
                    theta = loss / (2 * (np.linalg.norm(x) ** 2))
                    w[int(y[0]), :] = w[int(y[0]), :] + theta * x
                    w[y_hat, :] = w[y_hat, :] - theta * x
        return w
