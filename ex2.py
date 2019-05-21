import numpy as np
import Perceptron
import SVM
import PA
import utilis
import sys
import norm

BIAS = 53
MAX_RANGE = 3
MIN_RANGE = -2
PERC_EPOCHS = 13
PERC_RATE = 0.01
PA_EPOCHS = 22
SVM_EPOCHS = 10
SVM_LAMBDA = 0.01
SVM_RATE = 0.01
SEED_RANGE = 1000

"""
The main function, from here we start the operation
We initialize the weights, read to the sets and run the algorithms with the normalization
:param args: the arguments to the program
"""


def main(args):
    training_set = utilis.readExamples(args[1])
    training_labels_set = utilis.readLabelSet(args[2])
    init_weights = np.zeros((3, training_set.shape[1]))

    test_set = utilis.readExamples(args[3])
    norm_tra_set, norm_tes_set = norm.Min_Max_normalization(training_set, test_set, MAX_RANGE, MIN_RANGE)
    seed = utilis.generate_Seed(SEED_RANGE, BIAS)

    perc_obj = Perceptron.Perceptron(PERC_EPOCHS, PERC_RATE, norm_tra_set, training_labels_set, seed)
    perceptron_weights = perc_obj.training(None, None, init_weights)

    pa_obj = PA.PA(PA_EPOCHS, norm_tra_set, training_labels_set, seed)
    pa_weights = pa_obj.training(None, None, init_weights)

    norm_tra_set, norm_tes_set = norm.Zscore_normalization(training_set, test_set)

    svm_obj = SVM.SVM(SVM_EPOCHS, SVM_LAMBDA, SVM_RATE, norm_tra_set, training_labels_set, seed)
    svm_weights = svm_obj.training(None, None, init_weights)

    utilis.printThePredictions(perceptron_weights, svm_weights, pa_weights, test_set)


if __name__ == "__main__":
    main(sys.argv)
