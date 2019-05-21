import numpy as np

"""
The min max normalization function
:param training_set: the given training set to be trained
:param test_set: the given test set
:param max_range: the given max range
:param min_range: the given min range
"""


def Min_Max_normalization(training_set, test_set, max_range, min_range):
    norm_tra_set = training_set.copy()
    norm_tes_set = test_set.copy()
    for indx in range(len(norm_tra_set[0])):
        max = np.amax(norm_tra_set[:, indx])
        min = np.amin(norm_tra_set[:, indx])
        # check if not zero
        if min != max:
            norm_tra_set[:, indx] = ((norm_tra_set[:, indx] - min) / max - min) * (max_range - min_range) + min_range
            # check if idx not cross the test set limits
            if (indx < len(norm_tes_set)):
                norm_tes_set[:, indx] = ((norm_tes_set[:, indx] - min) / max - min) * (
                        max_range - min_range) + min_range
    return norm_tra_set, norm_tes_set


"""
The z-score normalization function
:param training_set: the given training set to be trained
:param test_set: the given test set
"""


def Zscore_normalization(training_set, test_set):
    norm_tra_set = training_set.copy()
    norm_tes_set = test_set.copy()
    for i in range(len(norm_tra_set[0])):
        mean = np.mean(norm_tra_set[:, i])
        std = np.std(norm_tra_set[:, i])
        if (std != 0):
            norm_tra_set[:, i] = (norm_tra_set[:, i] - mean) / std
            if (i < len(norm_tes_set)):
                norm_tes_set[:, i] = (norm_tes_set[:, i] - mean) / std
    return norm_tra_set, norm_tes_set
