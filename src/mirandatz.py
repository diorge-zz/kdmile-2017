import pandas as pd
import numpy as np
import os
import pickle
# import sklearn.metrics.mutual_info_score as rand_index
from sklearn.cluster import *

# Constants
DATASET_DIR = r"../data/matrices/"
LABELS_DIR = r"../data/labels/"
MATRICES_INDEXES = [46, 271, 368]


def main():
    datasets = acquire_datasets()
    labels = acquire_labels()

    for matrix_index in MATRICES_INDEXES:
        print(datasets[matrix_index].shape)
        print(labels[matrix_index].shape)
        print()


def acquire_datasets():
    datasets = {}
    for index in MATRICES_INDEXES:
        filename = f'{DATASET_DIR}{index}'
        matrix = np.genfromtxt(fname=filename)
        datasets[index] = matrix

    return datasets


def acquire_labels():
    labels = {}
    for index in MATRICES_INDEXES:
        filename = f'{LABELS_DIR}{index}.csv'
        temp = pd.read_csv(filepath_or_buffer=filename)
        labels[index] = temp

    return labels


if __name__ == "__main__":
    main()
    print("DONE!")
