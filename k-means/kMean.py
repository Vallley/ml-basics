import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from networkx.utils.misc import groups
from scipy.spatial.distance import cdist
import seaborn as sns
import random
from tqdm.notebook import tqdm
#from skimage.io import imread

#%matplotlib inline

# Зафиксируем случайность, чтобы у нас получались одинаковые результаты.
'''np.random.seed(seed=63)

p1 = np.random.normal(loc=0, scale=1, size=(50, 2))
p2 = np.random.normal(loc=5, scale=2, size=(50, 2))
p3 = np.random.normal(loc=10, scale=0.8, size=(50, 2)) - np.array([5, -5])

X = np.concatenate((p1, p2, p3))

plt.scatter(p1[:, 0], p1[:, 1])
plt.scatter(p2[:, 0], p2[:, 1])
plt.scatter(p3[:, 0], p3[:, 1])


plt.scatter(X[:, 0], X[:, 1])'''

def kmeans_predict(x, clusters):  #рассчитывает расстояния от центров кластеров до каждой точки и возвращает, к какому кластеру принадлежит каждая точка
    labels = []
    centroids_list = set()
    for xi in x:
        distances = np.zeros(len(clusters))
        for cluster in range(len(clusters)):
            dist = 0
            for i in range(len(xi)):
                dist += (clusters[cluster][i] - xi[i]) ** 2
            distances[cluster] = dist
        index = np.argmin(distances)
        labels.append(index)
        centroids_list.add(tuple(clusters[index]))
    return np.array(labels), np.array(centroids_list)


def kmeans_predict2(x, clusters):  # аналогично kmeans_predict
    labels = []
    centroids_list = set()
    distances = cdist(x, clusters, metric='euclidean')
    for distance in distances:
        index = np.argmin(distance)
        labels.append(index)
        centroids_list.add(tuple(clusters[index]))
    return np.array(labels), np.array(centroids_list)


def center_update(centroids, x):
    cent_history = []  # История центров кластеров
    cent_history.append(centroids)
    x = np.array(x)

    new_centroids = []
    clusters = []

    for j in range(len(centroids)):
        cluster = [c for c in x if int(c[-1]) == j]
        clusters.append(cluster)
    for cluster in clusters:
        center = np.sum(cluster, axis=0) / len(cluster)
        new_centroids.append(center[:-1])
    new_centroids = np.array(new_centroids)
    cent_history.append(new_centroids)
    #lables = kmeans_predict2(X, new_centroids)
    #x = np.hstack((X, lables.reshape(-1, 1)))

    return new_centroids, cent_history


def kmeans_fit_predict(x, k=8, max_iter=100, tol=0.001, low=0.0, high=1.0):
    centroids = np.random.uniform(low=low, high=high, size=(k, x.shape[1]))
    x = np.array(x)
    labels, centroids_list = kmeans_predict2(x, centroids)
    while len(set(labels)) < k:
        for c in range(len(centroids)):
            if centroids[c] not in centroids_list:
                centroids[c] = np.random.uniform(low=low, high=high, size=x.shape[1])
        labels, centroids_list = kmeans_predict2(x, centroids)
    x_with_lables = np.hstack((x, labels.reshape(-1, 1)))

    for _ in range(max_iter):
        new_centroids, center_history = center_update(centroids, x_with_lables)
        print((np.mean(centroids - new_centroids)) ** 2)
        if (np.mean(new_centroids - centroids)) ** 2 < tol:
            return centroids, x_with_lables, center_history
        else:
            diff = (np.mean(new_centroids - centroids)) ** 2
            print(f"Iteration {_}: mean diff = {diff:.6f}")
            centroids = new_centroids
            labels, centroids_list = kmeans_predict2(x, centroids)
            x_with_lables = np.hstack((x, labels.reshape(-1, 1)))
    return centroids, x_with_lables, center_history


# установим число кластеров k равное трем
# не генерируем центр кластера выше максимального значения из Х - ограничим это используя high
'''clusters_mnist, labels_mnist, cent_history = kmeans_fit_predict(X, k=3, low=0.0, high=np.max(X))

STEPS = len(cent_history) - 1  # количество шагов обновления центров

plt.figure(figsize=(10, 8))
for i in range(STEPS):
    plt.subplot((STEPS + 1) // 2, (STEPS + 1) // 2, i + 1)
    plt.plot(X[labels_mnist[:, 2] == 0, 0], X[labels_mnist[:, 2] == 0, 1], 'bo', label='cluster #1')
    plt.plot(X[labels_mnist[:, 2] == 1, 0], X[labels_mnist[:, 2] == 1, 1], 'co', label='cluster #2')
    plt.plot(X[labels_mnist[:, 2] == 2, 0], X[labels_mnist[:, 2] == 2, 1], 'mo', label='cluster #3')
    plt.plot(cent_history[i][:, 0], cent_history[i][:, 1], 'rX')
    plt.legend(loc=0)

'''




