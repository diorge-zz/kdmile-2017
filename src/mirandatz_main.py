import pandas as pd
import numpy as np
import os
import pickle
from miranda_run_algos import *
import os

# Constants
DATASET_DIR = r"../data/matrices/"
LABELS_DIR = r"../data/labels/"
MATRICES_INDEXES = [46, 271, 368]
COLUMN_DELIMITER = "\t"
OUTPUT_FILE = None


def main():
    datasets = acquire_datasets()
    labels = acquire_labels()

    # for _ in range(100):
    for matrix_index in reversed(MATRICES_INDEXES):
        X = datasets[matrix_index]
        y = labels[matrix_index]['label']
        # with open(f'out_{matrix_index}.csv', mode='a', encoding='utf8') as file:
        with open(os.devnull, mode='a', encoding='utf8') as file:
            # Writing header
            file.write(f'method_name{COLUMN_DELIMITER}score{COLUMN_DELIMITER}params\n')

            # Running all the clustering algorithms
            print(f'Processing dataset: {matrix_index}')
            print(f'Dataset shape: {X.shape}')
            pipeline(X=X, y=y, output_channel=file)
            print()
            file.write('\n')


def pipeline(X, y, output_channel):
    fiveMeansLabels = KMeans(n_clusters=5, init='k-means++', n_init=50, max_iter=300).fit_predict(X, None)
    fiveMeansScore = calculate_score(labels_true=y, labels_pred=fiveMeansLabels)
    print(f'5-Means: {fiveMeansScore}')
    output_channel.write(f'five_means{COLUMN_DELIMITER}{fiveMeansScore}{COLUMN_DELIMITER}k=5\n')

    print_score_and_params(fiveMeansScore=fiveMeansScore,
                           output_channel=output_channel,
                           method_name='RandomLabels',
                           score_and_params=
                           [
                               calculate_score(y, np.random.random_integers(low=0, high=X.shape[0], size=len(y))),
                               'np.random.random_integers(low=0, high=X.shape[0], size=len(y))'
                           ]
                           )

    print_score_and_params(fiveMeansScore=fiveMeansScore,
                           output_channel=output_channel,
                           method_name='KMeans',
                           score_and_params=run_kmeans(X, y))

    # print_score_and_params(fiveMeansScore=fiveMeansScore,
    #                        output_channel=output_channel,
    #                        method_name='Birch',
    #                        score_and_params=run_birch(X, y))

    print_score_and_params(fiveMeansScore=fiveMeansScore,
                           output_channel=output_channel,
                           method_name='AgglomerativeClustering',
                           score_and_params=run_agglomerative_clustering(X, y))

    print_score_and_params(fiveMeansScore=fiveMeansScore,
                           output_channel=output_channel,
                           method_name='DBScan',
                           score_and_params=run_dbscan(X, y))

    print_score_and_params(fiveMeansScore=fiveMeansScore,
                           output_channel=output_channel,
                           method_name='MeanShift',
                           score_and_params=run_mean_shift(X, y))

    print_score_and_params(fiveMeansScore=fiveMeansScore,
                           output_channel=output_channel,
                           method_name='AffinityPropagation',
                           score_and_params=run_affinity_propagation(X, y))


def print_score_and_params(fiveMeansScore, method_name, score_and_params, output_channel):
    score, params = score_and_params
    output_channel.write(f'{method_name}{COLUMN_DELIMITER}{score}{COLUMN_DELIMITER}{params}\n')
    if (score > fiveMeansScore):
        print(f'+++ {method_name} | score: {score} | {params}')
    else:
        print(f'--- {method_name} | score: {score} | {params}')


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
