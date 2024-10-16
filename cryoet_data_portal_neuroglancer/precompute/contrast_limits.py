"""Methods for computing contrast limits for Neuroglancer image layers."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate, find_peaks
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from cryoet_data_portal_neuroglancer.utils import ParameterOptimizer

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


def euclidean_distance(x: tuple[float, float], y: tuple[float, float]) -> float:
    return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


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
    def compute_contrast_limit(
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

    def optimize(self, real_limits, max_evals=100, loss_threshold=None, **kwargs) -> dict:
        """Hyper-parameter optimisation.

        Sub-classes should implement the
        _objective_function and _define_parameter_space methods.

        Parameters
        ----------
        params: dict
            The parameters for the optimisation.
            Keys are "low_percentile" and "high_percentile".
        real_limits: tuple[float, float]
            The real contrast limits.
        max_evals: int, optional.
            The maximum number of evaluations.
            By default 100.
        loss_threshold: float, optional.
            The loss threshold.
            By default None.
        **kwargs
            Additional keyword arguments - passed to hyperopt fmin.

        Returns
        -------
        dict, tuple[float, float]
            The best parameters found, and the contrast limits.
        """

        def _objective(params):
            return self.objective_function(params, real_limits)

        parameter_optimizer = ParameterOptimizer(_objective)
        self._define_parameter_space(parameter_optimizer)
        best = parameter_optimizer.optimize(
            max_evals=max_evals,
            loss_threshold=loss_threshold,
            **kwargs,
        )
        contrast_limits = self._objective_function(best)
        return best, contrast_limits

    def objective_function(self, params, real_limits):
        return euclidean_distance(
            self._objective_function(params),
            real_limits,
        )

    def _objective_function(self, params):
        return self.compute_contrast_limit(params["low_percentile"], params["high_percentile"])

    def _define_parameter_space(self, parameter_optimizer: ParameterOptimizer):
        parameter_optimizer.space_creator(
            {
                "low_percentile": {"type": "randint", "args": [0, 50]},
                "high_percentile": {"type": "randint", "args": [51, 100]},
            },
        )

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

    @compute_with_timer
    def compute_contrast_limit(
        self,
        num_components: int = 3,
        low_variance_mult: float = 3.0,
        high_variance_mult: float = 0.5,
    ) -> tuple[float, float]:
        """Calculate the contrast limits using Gaussian Mixture Model.

        Parameters
        ----------
        num_components: int, optional.
            The number of components to use for the GMM.
            By default 3.
        low_variance_mult: float, optional.
            The multiplier for the low variance.
            By default 3.0.
        high_variance_mult: float, optional.
            The multiplier for the high variance.
            By default 0.5.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        if num_components < 2:
            raise ValueError("Number of components must be at least 2.")
        self.num_components = num_components
        covariance_type = "full"
        self.gmm_estimator = GaussianMixture(
            n_components=num_components,
            covariance_type=covariance_type,
            max_iter=20,
            random_state=42,
            reg_covar=1e-5,
            init_params="k-means++",
        )

        sample_data = self.volume.flatten()
        try:
            self.gmm_estimator.fit(sample_data.reshape(-1, 1))
        except ValueError:
            # GMM fit can fail if the data is not well distributed - try with less components
            return self.compute_contrast_limit(
                num_components - 1,
                low_variance_mult,
                high_variance_mult,
            )

        # Get the stats for the gaussian which sits in the middle
        means = self.gmm_estimator.means_.flatten()
        covariances = self.gmm_estimator.covariances_

        # The shape depends on `covariance_type`::
        # (n_components,)                        if 'spherical',
        # (n_features, n_features)               if 'tied',
        # (n_components, n_features)             if 'diag',
        # (n_components, n_features, n_features) if 'full'
        variances = covariances.flatten()

        # Pick the GMM component which is closest to the mean of the volume
        volume_mean = np.mean(sample_data)
        closest_mean_index = np.argmin(np.abs(means - volume_mean))
        mean_to_use = means[closest_mean_index]
        std_to_use = np.sqrt(variances[closest_mean_index])

        return mean_to_use - low_variance_mult * std_to_use, mean_to_use + high_variance_mult * std_to_use

    def _objective_function(self, params):
        return self.compute_contrast_limit(
            params["num_components"],
            params["low_variance_mult"],
            params["high_variance_mult"],
        )

    def _define_parameter_space(self, parameter_optimizer):
        parameter_optimizer.space_creator(
            {
                "num_components": {"type": "randint", "args": [2, 4]},
                "low_variance_mult": {"type": "uniform", "args": [1.0, 5.0]},
                "high_variance_mult": {"type": "uniform", "args": [0.1, 1.0]},
            },
        )

    def plot(self, output_filename: Optional[str | Path] = None) -> None:
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
    def compute_contrast_limit(
        self,
        start_gradient: float = 0.08,
        end_gradient: float = 0.08,
        start_multiplier: float = 1.0,
        end_multiplier: float = 0.4,
    ) -> tuple[float, float]:
        """Calculate the contrast limits using the Cumulative Distribution Function.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        # Calculate the histogram of the volume
        n_bins = 512
        min_value = np.min(self.volume.flatten())
        max_value = np.max(self.volume.flatten())
        hist, bin_edges = np.histogram(self.volume.flatten(), bins=n_bins, range=[min_value, max_value])

        # Calculate the CDF of the histogram
        cdf = np.cumsum(hist) / np.sum(hist)

        # Find where the function starts to flatten
        try:
            gradient = np.gradient(cdf.compute())
        except AttributeError:
            gradient = np.gradient(cdf)
        largest_peak = np.argmax(gradient)
        peak_gradient_value = gradient[largest_peak]

        start_of_rising = np.where(gradient > start_gradient * peak_gradient_value)[0][0]
        # Find the first point after the largest peak where the gradient is less than 0.1 * peak_gradient_value
        end_of_flattening = np.where(gradient[largest_peak:] < end_gradient * peak_gradient_value)[0][0]
        end_of_flattening += largest_peak

        start_value = bin_edges[start_of_rising]
        end_value = bin_edges[end_of_flattening]
        middle_value = bin_edges[largest_peak]
        start_to_middle = middle_value - start_value
        middle_to_end = end_value - middle_value
        start_limit = middle_value - start_multiplier * start_to_middle
        end_limit = middle_value + end_multiplier * middle_to_end

        x = np.linspace(min_value, max_value, n_bins)
        self.cdf = [x, cdf]
        try:
            self.limits = (start_limit.compute(), end_limit.compute())
        except AttributeError:
            self.limits = (start_limit, end_limit)
        self.first_derivative = gradient
        self.second_derivative = np.gradient(gradient)

        return self.limits

    def _objective_function(self, params):
        return self.compute_contrast_limit(
            params["start_gradient"],
            params["end_gradient"],
            params["start_multiplier"],
            params["end_multiplier"],
        )

    def _define_parameter_space(self, parameter_optimizer):
        parameter_optimizer.space_creator(
            {
                "start_gradient": {"type": "uniform", "args": [0.01, 0.2]},
                "end_gradient": {"type": "uniform", "args": [0.01, 0.2]},
                "start_multiplier": {"type": "uniform", "args": [0.1, 1.5]},
                "end_multiplier": {"type": "uniform", "args": [0.1, 1.5]},
            },
        )

    def plot(self, output_filename: Optional[str | Path] = None, real_limits: Optional[list] = None) -> None:
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


class SignalDecimationContrastLimitCalculator(ContrastLimitCalculator):

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
        self.decimation = None

    @compute_with_timer
    def compute_contrast_limit(
        self,
        downsample_factor: int = 5,
        sample_factor: float = 0.10,
        threshold_factor: float = 0.01,
    ) -> tuple[float, float]:
        """Calculate the contrast limits using decimation.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        # Calculate the histogram of the volume
        n_bins = 512
        min_value = np.min(self.volume.flatten())
        max_value = np.max(self.volume.flatten())
        hist, _ = np.histogram(self.volume.flatten(), bins=n_bins, range=[min_value, max_value])

        # Calculate the CDF of the histogram
        cdf = np.cumsum(hist) / np.sum(hist)
        x = np.linspace(min_value, max_value, n_bins)

        # Downsampling the CDF
        y_decimated = decimate(cdf, downsample_factor)
        x_decimated = np.linspace(np.min(x), np.max(x), len(y_decimated))

        # Calculate the absolute differences between consecutive points in the decimated CDF
        diff_decimated = np.abs(np.diff(y_decimated))

        # Compute threshold and lower_change threshold
        sample_size = int(sample_factor * len(diff_decimated))

        initial_flat = np.mean(cdf[:sample_size])  # Average of first points (assumed flat region)
        final_flat = np.mean(cdf[-sample_size:])  # Average of last points (assumed flat region)
        midpoint = (initial_flat + final_flat) / 2
        curve_threshold = threshold_factor * midpoint

        # Detect start and end of slope
        start_idx_decimated = np.argmax(diff_decimated > curve_threshold)  # First large change
        end_idx_decimated = (
            np.argmax(diff_decimated[start_idx_decimated + 1 :] < curve_threshold) + start_idx_decimated
        )  # first small change

        # Map back the indices to original values
        self.cdf = [x, cdf]
        self.limits = (
            (x_decimated[start_idx_decimated], x_decimated[end_idx_decimated])
            if end_idx_decimated != -1
            else (None, None)
        )

        return self.limits

    def _objective_function(self, params):
        return self.compute_contrast_limit(
            params["downsample_factor"],
            params["sample_factor"],
            params["threshold_factor"],
        )

    def _define_parameter_space(self, parameter_optimizer):
        parameter_optimizer.space_creator(
            {
                "downsample_factor": {"type": "randint", "args": [3, 7]},
                "sample_factor": {"type": "uniform", "args": [0.01, 0.1]},
                "threshold_factor": {"type": "uniform", "args": [0.005, 0.2]},
            },
        )

    def plot(self, output_filename: Optional[str | Path] = None, real_limits: Optional[list] = None) -> None:
        """Plot the CDF and the calculated limits."""
        fig, ax = plt.subplots()

        ax.plot(self.cdf[0], self.cdf[1])
        ax.axvline(self.limits[0], color="r")
        ax.axvline(self.limits[1], color="r")

        if real_limits:
            ax.axvline(real_limits[0], color="b")
            ax.axvline(real_limits[1], color="b")

        if output_filename:
            fig.savefig(output_filename)
        else:
            plt.show()
        plt.close(fig)


def combined_contrast_limit_plot(
    cdf: list[list[float], list[float]],
    real_limits: tuple[float, float],
    limits_dict: dict[str, tuple[float, float]],
    output_filename: Optional[str | Path] = None,
) -> None:
    """Plot the CDF and the calculated limits."""
    fig, ax = plt.subplots()

    ax.plot(cdf[0], cdf[1])
    ax.axvline(real_limits[0], color="b")
    ax.axvline(real_limits[1], color="b")

    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color="b", lw=4)]
    colors_dict = {"gmm": "g", "cdf": "y", "decimation": "r"}
    min_x = real_limits[0]
    max_x = real_limits[1]
    for key, limits in limits_dict.items():
        min_x = min(min_x, limits[0])
        max_x = max(max_x, limits[1])
        color = colors_dict.get(key, "k")
        ax.axvline(limits[0], color=color)
        ax.axvline(limits[1], color=color)
        custom_lines.append(Line2D([0], [0], color=color, lw=4))

    ax.set_xlim(min_x, max_x)

    # Produce a legend
    legend = ["Real Limits"]
    for key in limits_dict:
        legend.append(key + " Limits")
    ax.legend(custom_lines, legend)

    if output_filename:
        fig.savefig(output_filename)
    else:
        plt.show()
    plt.close(fig)
