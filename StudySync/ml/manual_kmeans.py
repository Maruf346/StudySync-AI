# StudySync/ml/manual_kmeans.py
import numpy as np
import random

class ManualKMeans:
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)

        # Randomly choose initial centroids
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iter):
            # Assign labels
            labels = self.predict(X)

            # Recalculate centroids
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            # Check convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
