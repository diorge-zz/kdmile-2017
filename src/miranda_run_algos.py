import numpy as np
from sklearn.cluster import *
from sklearn.metrics import make_scorer
from sklearn.metrics import adjusted_rand_score


def run_kmeans(X, y):
    element_count, feature_count = X.shape
    scores_and_params = []

    for k in np.arange(2, element_count):
        labels_pred = KMeans(n_clusters=k, init='k-means++', n_init=50) \
            .fit_predict(X, None)
        score = calculate_score(labels_true=y, labels_pred=labels_pred)
        scores_and_params.append((score, f'k: {k}'))

    return get_max_score_and_params(scores_and_params)


def run_agglomerative_clustering(X, y):
    element_count, feature_count = X.shape
    scores_and_params = []

    #for n_clusters in np.arange(2, element_count):
    for n_clusters in [15]:
        for linkage in ["ward", "complete", "average"]:
            labels_pred = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage) \
                .fit_predict(X, None)
            score = calculate_score(labels_true=y, labels_pred=labels_pred)
            scores_and_params.append((score, f'n_cluster: {n_clusters}, linkage: {linkage}'))

    return get_max_score_and_params(scores_and_params)


def run_dbscan(X, y):
    element_count, feature_count = X.shape
    scores_and_params = []

    for min_samples in np.arange(2, element_count):
        for eps in np.linspace(0.01, 0.95, 100):
            labels_pred = DBSCAN(eps=eps, min_samples=min_samples) \
                .fit_predict(X, None)
            score = calculate_score(labels_true=y, labels_pred=labels_pred)
            scores_and_params.append((score, f'min_samples: {min_samples}, eps: {eps}'))

    return get_max_score_and_params(scores_and_params)


def run_mean_shift(X, y):
    element_count, feature_count = X.shape
    scores_and_params = []

    for bandwidth in np.linspace(0.05, 0.95, 50):
        labels_pred = MeanShift(bandwidth=bandwidth) \
            .fit_predict(X, None)
        score = calculate_score(labels_true=y, labels_pred=labels_pred)
        scores_and_params.append((score, f'bandwidth: {bandwidth}'))

    return get_max_score_and_params(scores_and_params)


def run_affinity_propagation(X, y):
    element_count, feature_count = X.shape
    scores_and_params = []

    preferences = []
    for i in range(2, element_count - 1):
        temp = np.zeros(element_count)
        for j in range(i):
            temp[j] = 100
        preferences.append(temp)

    for damping in np.linspace(start=0.5, stop=0.95, num=50):
        for preference in preferences:
            labels_pred = AffinityPropagation(damping=damping, preference=preference) \
                .fit_predict(X, None)
            score = calculate_score(labels_true=y, labels_pred=labels_pred)
            scores_and_params.append((score, f'damping: {damping}, preference: {preference}'))

    return get_max_score_and_params(scores_and_params)


def run_birch(X, y):
    element_count, feature_count = X.shape
    scores_and_params = []

    for n_clusters in np.arange(2, element_count):
        for branching_factor in np.arange(2, element_count):
            for threshold in np.linspace(start=0.5, stop=0.95, num=50):
                labels_pred = Birch(n_clusters=int(n_clusters), branching_factor=branching_factor, threshold=threshold) \
                    .fit_predict(X, None)
                score = calculate_score(labels_true=y, labels_pred=labels_pred)
                params = f'n_clusters: {n_clusters},' \
                         f' branching_factor: {branching_factor},' \
                         f' threshold: {threshold}'
            scores_and_params.append((score, params))

    return get_max_score_and_params(scores_and_params)


def get_max_score_and_params(scores_and_params):
    max_score = None
    max_params = None
    for score, params in scores_and_params:
        if max_score == None or max_score < score:
            max_score = score
            max_params = params
    return max_score, max_params


def calculate_score(labels_true, labels_pred):
    score = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)
    # Normalizing
    # score = (score + 1) / 2
    return score
