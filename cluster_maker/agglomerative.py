from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _compute_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Compute simple centroids as the mean of points in each cluster.
    Useful for plotting and inertia-style metrics even though
    agglomerative clustering does not optimise centroid distance directly.
    """
    centroids = np.zeros((k, X.shape[1]), dtype=float)
    for cid in range(k):
        mask = labels == cid
        if not np.any(mask):
            # Fallback: if cluster is empty, pick a random point
            centroids[cid] = X[np.random.randint(0, X.shape[0])]
        else:
            centroids[cid] = X[mask].mean(axis=0)
    return centroids


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run hierarchical agglomerative clustering on a numeric matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
    metric : str, default "euclidean"
        Distance metric used by the linkage (ignored when linkage="ward").

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean of points in each cluster, provided for plotting/metrics.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if k <= 1:
        raise ValueError("k must be at least 2 for agglomerative clustering.")

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric if linkage != "ward" else "euclidean",
    )
    labels = model.fit_predict(X)
    centroids = _compute_centroids(X, labels, k)
    return labels, centroids
