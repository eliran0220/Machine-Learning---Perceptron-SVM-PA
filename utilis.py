import numpy as np
import csv

"""
The function generates the array seed based on the bias parameter
:param amount: the range
:param bias: the given bias
"""


def generate_Seed(amount, bias):
    array_seed = []
    for i in range(amount):
        array_seed.append(i * bias)
    return array_seed


"""
The function prints the prediction of each algorithm
:param perceptron_weights: the weights of perceptron
:param svm_weights: the weights of perceptron
:param pa_weights: the weights of perceptron
:param test_set: the given test set to compare with
"""


def printThePredictions(perceptron_weights, svm_weights, pa_weights, test_set):
    for i in range(len(test_set)):
        x = test_set[i]
        y_hat_perceptron = np.argmax(np.dot(perceptron_weights, x))
        y_hat_svm = np.argmax(np.dot(svm_weights, x))
        y_hat_pa = np.argmax(np.dot(pa_weights, x))
        print("perceptron: " + str(int(y_hat_perceptron)) + ", svm: " + str(int(y_hat_svm)) + ", pa: " + str(
            int(y_hat_pa)))


"""
The function reads the label set by given path
:param filePath: the path of the given file to read from
"""


def readLabelSet(filePath):
    label_set_file = open(filePath)
    label_set = []
    for line in label_set_file:
        temp = [line]
        label_set.append(temp)
    label_set_file.close()
    return np.array(label_set, "float")


"""
The function reads the training set to be trained in the algorithms
:param filePath: the path of the given file to read from
"""


def readExamples(filePath):
    traning_set_file = open(filePath)
    csv_file = csv.reader(traning_set_file)
    traning_set = []
    # here we are going to use hot encoding, depends on the char
    for row in csv_file:
        temp = []
        if row[0] == 'M':
            temp.append(1)
            temp.append(0)
            temp.append(0)
        elif row[0] == 'F':
            temp.append(0)
            temp.append(1)
            temp.append(0)
        elif row[0] == 'I':
            temp.append(0)
            temp.append(0)
            temp.append(1)
        for i in range(1, 8):
            temp.append(float(row[i]))
        traning_set.append(temp)
    traning_set_file.close()
    return np.array(traning_set)
