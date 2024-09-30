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
        LOGGER.info(f"Running function {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        LOGGER.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
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
        lowest_points, _ = find_peaks(-standard_deviation_per_z_slice, prominence=0.05)
        if len(lowest_points) < 2:
            LOGGER.warning("Could not find enough low points in the standard deviation per z-slice.")
            return volume
        for value in lowest_points:
            if value < central_z_slice:
                z_min = value
            else:
                z_max = min(volume.shape[0], value + 1)
                break

    else:
        z_min = max(0, int(np.ceil(central_z_slice - z_radius)))
        z_max = min(volume.shape[0], int(np.floor(central_z_slice + z_radius) + 1))
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

    def take_random_samples_from_volume(self, num_samples: int = 100_000) -> None:
        """Take random samples from the volume.

        Parameters
        ----------
            num_samples: int
                The number of samples to take.

        Returns
        -------
            np.ndarray
                The random samples.
        """
        generator = np.random.default_rng(0)
        if len(self.volume.flatten()) > num_samples:
            self.volume = generator.choice(
                self.volume.flatten(),
                num_samples,
                replace=False,
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

        try:
            return low_value.compute()[0], high_value.compute()[0]
        except AttributeError:
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
        sample_data = self.volume.flatten()
        self.gmm_estimator.fit(sample_data.reshape(-1, 1))

        # Get the stats for the gaussian which sits in the middle
        means = self.gmm_estimator.means_.flatten()
        covariances = self.gmm_estimator.covariances_.flatten()

        # pick the middle GMM component - TODO should actually be the one with the
        # mean closest to the mean of the volume
        volume_mean = np.mean(sample_data)
        closest_mean_index = np.argmin(np.abs(means - volume_mean))
        mean_to_use = means[closest_mean_index]
        covariance_to_use = covariances[closest_mean_index]
        variance_to_use = np.sqrt(covariance_to_use)

        return mean_to_use - 3 * variance_to_use, mean_to_use + 0.5 * variance_to_use

    def plot_gmm_clusters(self, output_filename: Optional[str | Path] = None) -> None:
        """Plot the GMM clusters."""
        fig, ax = plt.subplots()

        ax.plot(
            self.gmm_estimator.means_.flatten(),
            [np.sqrt(y) for y in self.gmm_estimator.covariances_.flatten()],
            "o",
        )
        ax.set_xlabel("Mean")
        ax.set_ylabel("Standard Deviation")
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

        ax.plot(self.kmeans_estimator.cluster_centers_, "o")
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
        sample_data = self.volume.flatten()
        self.kmeans_estimator.fit(sample_data.reshape(-1, 1))

        cluster_centers = self.kmeans_estimator.cluster_centers_.flatten()

        # Find the cluster which is closest to the mean of the volume
        volume_mean = np.mean(sample_data)
        closest_cluster_index = np.argmin(np.abs(cluster_centers - volume_mean))

        # Find the closest cluster to that mean cluster
        closest_distance = None
        for i in range(0, self.num_clusters):
            if i == closest_cluster_index:
                continue
            distance = np.abs(cluster_centers[i] - cluster_centers[closest_cluster_index])
            if closest_distance is None or distance < closest_distance:
                closest_cluster_index = i
                closest_distance = distance

        left_boundary = cluster_centers[closest_cluster_index] - 0.1 * closest_distance
        right_boundary = cluster_centers[closest_cluster_index] + 0.1 * closest_distance

        return left_boundary, right_boundary


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
        self.second_derivative = None

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
        hist, bin_edges = np.histogram(self.volume.flatten(), bins=400, range=[min_value, max_value])

        # Calculate the CDF of the histogram
        cdf = np.cumsum(hist) / np.sum(hist)

        # Find where the function starts to flatten
        try:
            gradient = np.gradient(cdf.compute())
        except AttributeError:
            gradient = np.gradient(cdf)
        second_derivative = np.gradient(gradient)
        peaks, _ = find_peaks(second_derivative, prominence=0.01)

        # If no peaks, take the argmax of the gradient
        biggest_peak = np.argmax(second_derivative) if len(peaks) == 0 else peaks[np.argmax(second_derivative[peaks])]

        negative_peaks, _ = find_peaks(-second_derivative, prominence=0.01)
        smallest_negative_peak = (
            np.argmin(second_derivative)
            if len(negative_peaks) == 0
            else negative_peaks[np.argmin(second_derivative[negative_peaks])]
        )

        x = np.linspace(min_value, max_value, 400)
        self.cdf = [x, cdf]
        try:
            self.limits = (
                bin_edges[biggest_peak].compute(),
                bin_edges[smallest_negative_peak].compute(),
            )
        except AttributeError:
            self.limits = (bin_edges[biggest_peak], bin_edges[smallest_negative_peak])
        self.first_derivative = gradient
        self.second_derivative = second_derivative

        # Shrink the limits a bit
        limit_width = self.limits[1] - self.limits[0]
        self.limits = (self.limits[0] + 0.1 * limit_width, self.limits[1] - 0.3 * limit_width)

        return self.limits

    def plot_cdf(self, output_filename: Optional[str | Path] = None, real_limits: Optional[list] = None) -> None:
        """Plot the CDF and the calculated limits."""
        fig, ax = plt.subplots()

        ax.plot(self.cdf[0], self.cdf[1])
        ax.axvline(self.limits[0], color="r")
        ax.axvline(self.limits[1], color="r")

        if real_limits:
            ax.axvline(real_limits[0], color="b")
            ax.axvline(real_limits[1], color="b")

        ax.plot(self.cdf[0], self.first_derivative * 100, "y")
        ax.plot(self.cdf[0], self.second_derivative * 100, "g")

        if output_filename:
            fig.savefig(output_filename)
        else:
            plt.show()
        plt.close(fig)
