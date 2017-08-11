import pandas as pd
import numpy as np
import os
import pickle
from miranda_run_algos import *

# Constants
TEST_DATASET_DIR = r"../data/test_sets/"
TEST_LABELS_DIR = r"../data/test_labels/"
TEST_DATASET_COUNT = 4
COLUMN_DELIMITER = ","


def main():
    datasets = acquire_datasets()
    labels = acquire_labels()

    for index in range(TEST_DATASET_COUNT):
        print(datasets[index].shape)
        print(labels[index].shape)
        print()
        X = datasets[index]
        y = labels[index]

        score = run_agglomerative_clustering(X=X, y=y)
        print(score)

def acquire_datasets():
    datasets = {}
    for index in range(TEST_DATASET_COUNT):
        filename = f'{TEST_DATASET_DIR}{index}.csv'
        matrix = np.genfromtxt(fname=filename, delimiter=COLUMN_DELIMITER)
        datasets[index] = matrix

    return datasets


def acquire_labels():
    labels = {}
    for index in range(TEST_DATASET_COUNT):
        filename = f'{TEST_LABELS_DIR}{index}.txt'
        temp = np.genfromtxt(filename, delimiter=COLUMN_DELIMITER)
        labels[index] = temp

    return labels


if __name__ == "__main__":
    main()
    print("DONE!")
