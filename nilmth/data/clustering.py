import itertools
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nilmth.utils.plot import plot_power_distribution
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist


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

    def perform_clustering(
        self, ser: np.array, method: Optional[str] = None
    ) -> np.array:
        """Performs the actual clustering, using the linkage function

        Parameters
        ----------
        ser : np.array
            Series of points to group in clusters
        method : str, optional
            Clustering method, by default None (takes the one from the class)

        Returns
        -------
        np.array
            Z[i] will tell us which clusters were merged in the i-th iteration
        """
        self.method = method if method is not None else self.method
        # The shape of our X matrix must be (n, m)
        # n = samples, m = features
        self.x = np.expand_dims(ser, axis=1)
        self.z = linkage(self.x, method=self.method)

    @property
    def cophenet(self):
        # Cophenet correlation coefficient
        c, coph_dists = cophenet(self.z, pdist(self.x))
        return c

    def plot_dendrogram(
        self, p: int = 6, max_d: Optional[float] = None, figsize: Tuple[int] = (3, 3)
    ):
        """Plots the dendrogram

        Parameters
        ----------
        p : int, optional
            Last split, by default 6
        max_d : Optional[float], optional
            Maximum distance between splits, by default None
        figsize : Tuple[int], optional
            Figure size, by default (3, 3)
        """
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

    def plot_dendrogram_distance(self, figsize: Tuple[int] = (10, 3)):
        """Plots the dendrogram distances

        Parameters
        ----------
        figsize : Tuple[int], optional
            Size of the figure, by default (10, 3)
        """
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
        self,
        n_cluster: Optional[int] = None,
        criterion: Optional[str] = None,
        centroid: str = "median",
    ):
        """Computes the thresholds and centroids of each group

        Parameters
        ----------
        n_cluster : Optional[int], optional
            Number of clusters, by default None
        criterion : Optional[str], optional
            Criterion used to compute the clusters, by default None
        centroid : str, optional
            Method to compute the centroids (median or mean), by default "median"
        """
        self.n_cluster = n_cluster if n_cluster is not None else self.n_cluster
        self.criterion = criterion if criterion is not None else self.criterion
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
