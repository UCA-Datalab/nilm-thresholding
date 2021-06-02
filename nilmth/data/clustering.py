import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist

from nilmth.utils.plot import plot_power_distribution


class HierarchicalClustering:
    x: np.array = None
    z: np.array = None
    thresh: np.array = None
    centroids: np.array = None
    dendrogram: dict = None

    def __init__(
        self, method: str = "average", n_cluster: int = 2, criterion: str = "maxclust"
    ):
        self.method = method
        self.n_cluster = n_cluster
        self.criterion = criterion

    def perform_clustering(self, ser: np.array, method: str = None):
        """Performs the actual clustering, using the linkage function

        Returns
        -------
        numpy.array
            Z[i] will tell us which clusters were merged in the i-th iteration
        """
        if method is not None:
            self.method = method
        # The shape of our X matrix must be (n, m)
        # n = samples, m = features
        self.x = np.expand_dims(ser, axis=1)
        self.z = linkage(self.x, method=self.method)

    @property
    def cophenet(self):
        # Cophenet correlation coefficient
        c, coph_dists = cophenet(self.z, pdist(self.x))
        return c

    def plot_dendrogram(self, p=6, max_d=None, figsize=(3, 3)):
        fig, ax = plt.subplots(figsize=figsize)
        self.dendrogram = dendrogram(
            self.z,
            p=p,
            orientation="right",
            truncate_mode="lastp",
            labels=self.x[:, 0],
            ax=ax,
        )
        if max_d is not None:
            ax.axvline(x=max_d, c="k")
        return fig, ax

    @property
    def dendrogram_distance(self):
        return sorted(set(itertools.chain(*self.dendrogram["dcoord"])), reverse=True)

    def plot_dendrogram_distance(self, figsize=(10, 3)):
        # Initialize plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        # Dendrogram distance
        ax1.scatter(
            range(2, len(self.dendrogram_distance) + 1), self.dendrogram_distance[:-1]
        )
        ax1.set_ylabel("Distance")
        ax1.set_xlabel("Number of clusters")
        ax1.grid()
        # Dendrogram distance difference
        diff = np.divide(
            -np.diff(self.dendrogram_distance), self.dendrogram_distance[:-1]
        )
        ax2.scatter(range(3, len(self.dendrogram_distance) + 1), diff[:-1])
        ax2.set_ylabel("Gradient")
        ax2.set_xlabel("Number of clusters")
        ax2.grid()
        return fig, (ax1, ax2)

    def compute_thresholds_and_centroids(
        self, n_cluster: int = None, criterion: str = None, centroid: str = "median"
    ):
        if n_cluster is not None:
            self.n_cluster = n_cluster
        if criterion is not None:
            self.criterion = criterion
        clusters = fcluster(self.z, self.n_cluster, self.criterion)
        # Get centroids
        if centroid == "median":
            fun = np.median
        elif centroid == "mean":
            fun = np.mean
        self.centroids = np.array(
            sorted([fun(self.x[clusters == (c + 1)]) for c in range(self.n_cluster)])
        )
        # Sort clusters by power
        x_max = sorted(
            [np.max(self.x[clusters == (c + 1)]) for c in range(self.n_cluster)]
        )
        x_min = sorted(
            [np.min(self.x[clusters == (c + 1)]) for c in range(self.n_cluster)]
        )
        self.thresh = np.divide(np.array(x_min[1:]) + np.array(x_max[:-1]), 2)

    def plot_cluster_distribution(self, label="", bins=100):
        fig, ax = plot_power_distribution(self.x, label, bins=bins)
        [ax.axvline(t, color="r", linestyle="--") for t in self.thresh]
        return fig, ax
