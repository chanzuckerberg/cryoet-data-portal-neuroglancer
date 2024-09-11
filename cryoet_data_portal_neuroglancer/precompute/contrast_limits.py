"""Methods for computing contrast limits for Neuroglancer image layers."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

LOGGER = logging.getLogger(__name__)


def compute_with_timer(func):
    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()
        LOGGER.info(f"Running function {func.__name__}.")
        result = func(*args, **kwargs)
        end_time = time.time()
        LOGGER.info(f"Function {func.__name__} took {end_time - start_time} seconds.")
        return result

    return wrapper


# TODO fix this to work with dask data, see the mesh changes for reference
def _restrict_volume_around_central_z_slice(
    volume: "np.ndarray",
    central_z_slice: Optional[int] = None,
    z_radius: Optional[int] = None,
) -> "np.ndarray":
    """Restrict a 3D volume to a region around a central z-slice.

    Parameters
    ----------
        volume: np.ndarray
            3D numpy array with Z as the first axis.
        central_z_slice: int or None, optional.
            The central z-slice around which to restrict the volume.
            By default None, in which case the central z-slice is the middle slice.
        z_radius: int, optional.
            The number of z-slices to include above and below the central z-slice.
            By default None, in which case it is auto computed.

    Returns
    -------
        np.ndarray
            3D numpy array. The restricted volume.
    """
    standard_deviation_per_z_slice = np.std(volume, axis=(1, 2))
    central_z_slice = central_z_slice or (0.5 * volume.shape[0] - 0.5)

    if z_radius is None:
        lowest_points = find_peaks(-standard_deviation_per_z_slice, prominence=0.1)[0]
        if len(lowest_points) < 2:
            raise ValueError("Not enough low points found to auto compute z-radius.")
        for value in lowest_points:
            if value < central_z_slice:
                z_min = value
            else:
                z_max = min(volume.shape[0], value + 1)
                break

    else:
        z_min = max(0, int(np.ceil(central_z_slice - z_radius)))
        z_max = min(volume.shape[0], int(np.floor(central_z_slice + z_radius) + 1))
    print(f"Z min: {z_min}, Z max: {z_max}")
    return volume[z_min:z_max]


class ContrastLimitCalculator:

    def __init__(self, volume: Optional["np.ndarray"] = None):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray or None, optional.
                Input volume for calculating contrast limits.
        """
        self.volume = volume

    def set_volume_and_z_limits(
        self,
        volume: "np.ndarray",
        central_z_slice: Optional[int] = None,
        z_radius: Optional[int] = None,
    ) -> None:
        """Set the volume and z-limits for calculating contrast limits.

        Parameters
        ----------
            volume: np.ndarray
                3D numpy array with Z as the first axis.
            central_z_slice: int or None, optional.
                The central z-slice around which to restrict the volume.
                By default None, in which case the central z-slice is the middle slice.
            z_radius: int or None, optional.
                The number of z-slices to include above and below the central z-slice.
                By default None.
        """
        self.volume = volume
        self.trim_volume_around_central_zslice(central_z_slice, z_radius)

    def trim_volume_around_central_zslice(
        self,
        central_z_slice: Optional[int] = None,
        z_radius: Optional[int] = None,
    ) -> None:
        """Trim the volume around a central z-slice.

        Parameters
        ----------
            central_z_slice: int or None, optional.
                The central z-slice around which to restrict the volume.
                By default None, in which case the central z-slice is the middle slice.
            z_radius: int or None, optional.
                The number of z-slices to include above and below the central z-slice.
                By default None, in which case it is auto computed.
        """
        self.volume = _restrict_volume_around_central_z_slice(
            self.volume,
            central_z_slice,
            z_radius,
        )

    @compute_with_timer
    def contrast_limits_from_percentiles(
        self,
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
    ) -> tuple[float, float]:
        """Calculate the contrast limits from the given percentiles.

        Parameters
        ----------
            low_percentile: float
                The low percentile for the contrast limit.
            high_percentile: float
                The high percentile for the contrast limit.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        low_value = np.percentile(self.volume.flatten(), low_percentile)
        high_value = np.percentile(self.volume.flatten(), high_percentile)

        return low_value, high_value

    @compute_with_timer
    def contrast_limits_from_mean(
        self,
        multipler: float = 3.0,
    ) -> tuple[float, float]:
        """Calculate the contrast limits from the mean and RMS.

        Parameters
        ----------
            multipler: float, optional.
                The multiplier for the RMS value.
                By default 3.0.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        mean_value = np.mean(self.volume)
        rms_value = np.sqrt(np.mean(self.volume**2))
        width = multipler * rms_value

        return mean_value - width, mean_value + width


class GMMContrastLimitCalculator(ContrastLimitCalculator):

    def __init__(self, volume: Optional["np.ndarray"] = None, num_components: int = 3):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray or None, optional.
                Input volume for calculating contrast limits.
            num_components: int, optional.
                The number of components to use for GMM.
                By default 3.
        """
        super().__init__(volume)
        self.num_components = num_components
        # cov_type in ["spherical", "diag", "tied", "full"]
        self.gmm_estimator = GaussianMixture(
            n_components=num_components,
            covariance_type="full",
            max_iter=20,
            random_state=0,
            init_params="k-means++",
        )

    @compute_with_timer
    def contrast_limits_from_gmm(self) -> tuple[float, float]:
        """Calculate the contrast limits using Gaussian Mixture Model.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        self.gmm_estimator.fit(self.volume.reshape(-1, 1))

        # Get the stats for the gaussian which sits in the middle
        means = self.gmm_estimator.means_.flatten()
        covariances = self.gmm_estimator.covariances_.flatten()

        return means[1] - 2 * covariances[1], means[1] + 2 * covariances[1]

    def plot_gmm_clusters(self, output_filename: Optional[str | Path] = None) -> None:
        """Plot the GMM clusters."""
        fig, ax = plt.subplots()

        # TODO improve this plot with std
        ax.plot(self.gmm_estimator.means_)
        if output_filename:
            fig.savefig(output_filename)
        else:
            plt.show()
        plt.close(fig)


class KMeansContrastLimitCalculator(ContrastLimitCalculator):

    def __init__(self, volume: Optional["np.ndarray"] = None, num_clusters: int = 3):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray or None, optional.
                Input volume for calculating contrast limits.
            num_clusters: int, optional.
                The number of clusters to use for KMeans.
                By default 3.
        """
        super().__init__(volume)
        self.num_clusters = num_clusters
        self.kmeans_estimator = KMeans(n_clusters=num_clusters, random_state=0)

    def plot_kmeans_clusters(self, output_filename: Optional[str | Path] = None) -> None:
        """Plot the KMeans clusters."""
        fig, ax = plt.subplots()

        ax.plot(self.kmeans_estimator.cluster_centers_)
        if output_filename:
            fig.savefig(output_filename)
        else:
            plt.show()
        plt.close(fig)

    @compute_with_timer
    def contrast_limits_from_kmeans(self) -> tuple[float, float]:
        """Calculate the contrast limits using KMeans clustering.

        Parameters
        ----------
            num_clusters: int, optional.
                The number of clusters to use for KMeans.
                By default 3.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        LOGGER.info("Calculating contrast limits from KMeans.")
        self.kmeans_estimator.fit(self.volume.reshape(-1, 1))

        cluster_centers = self.kmeans_estimator.cluster_centers_
        cluster_centers.sort()

        return cluster_centers[0], cluster_centers[-1]


class CDFContrastLimitCalculator(ContrastLimitCalculator):

    def __init__(self, volume: Optional["np.ndarray"] = None):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray or None, optional.
                Input volume for calculating contrast limits.
        """
        super().__init__(volume)
        self.cdf = None
        self.limits = None

    @compute_with_timer
    def contrast_limits_from_cdf(self) -> tuple[float, float]:
        """Calculate the contrast limits using the Cumulative Distribution Function.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        # Calculate the histogram of the volume
        min_value = np.min(self.volume.flatten())
        max_value = np.max(self.volume.flatten())
        hist, bin_edges = np.histogram(self.volume.flatten(), bins=1000, range=[min_value, max_value])

        # Calculate the CDF of the histogram
        cdf = np.cumsum(hist) / np.sum(hist)
        gradient = np.gradient(cdf)

        # Find the biggest positive peak
        peaks = find_peaks(gradient, prominence=0.1)
        biggest_peak = np.argmax(gradient[peaks])

        # Find where the function starts to become flat after the peak
        second_derivative = np.gradient(gradient)
        flat_points = np.where(second_derivative[biggest_peak:] < 0.0001)[0]
        # TODO improve error handling

        self.cdf = cdf
        self.limits = bin_edges[biggest_peak], bin_edges[biggest_peak + flat_points[0]]

        return self.limits

    def plot_cdf_and_limits(self, output_filename: Optional[str | Path] = None) -> None:
        """Plot the CDF and the calculated limits."""
        fig, ax = plt.subplots()

        ax.plot(self.cdf)
        ax.axvline(self.limits[0], color="r")
        ax.axvline(self.limits[1], color="r")

        if output_filename:
            fig.savefig(output_filename)
        else:
            plt.show()
        plt.close(fig)


# Other possibility is to take the derivative of the histogram and find the peaks
