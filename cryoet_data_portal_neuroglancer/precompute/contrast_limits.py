"""Methods for computing contrast limits for Neuroglancer image layers."""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture

from cryoet_data_portal_neuroglancer.utils import ParameterOptimizer

LOGGER = logging.getLogger(__name__)


def compute_contrast_limits(
    data: da.Array | np.ndarray,
    method: Literal["gmm", "cdf"] = "gmm",
    z_radius: int | None | Literal["auto", "computed"] = "auto",
    downsampling_ratio: float | None = 0.3,
) -> tuple[float, float]:
    """Compute the contrast limits for the given data.

    Parameters
    ----------
    data: da.Array or np.ndarray
        The input data for calculating contrast limits. Must be 3D.
    method: str, optional.
        The method to use for calculating contrast limits.
        By default "gmm". Other option is "cdf".
    z_radius: int or None or "auto", optional.
        The number of z-slices to include above and below the central z-slice.
        None means the whole volume is used.
        If "auto", the z-radius mode is picked based on the method.
        In "auto", the GMM method uses more of the volume, while the CDF method uses less.
        "compute" attempts to estimate a good z-slicing, but
        can currently be problematic for large volumes.
        default is "auto".
    downsampling_ratio: float or None, optional.
        The downsampling ratio for the volume. By default 0.3.
        If None, no downsampling is performed (same as 1.0).
        This is particularly useful if the z_radius is set to None.

    Returns
    -------
    tuple[float, float]
        The calculated contrast limits.
    """
    calculator_class: ContrastLimitCalculator = (
        GMMContrastLimitCalculator if method == "gmm" else CDFContrastLimitCalculator
    )
    if z_radius is not None and z_radius == "auto":
        z_radius = 15 if method == "gmm" else 5
    num_samples = None if downsampling_ratio is None else int(np.prod(data.shape) * downsampling_ratio)
    return calculator_class(data, z_radius=z_radius, num_samples=num_samples).compute_contrast_limit()


def _restrict_volume_around_central_z_slice(
    volume: np.ndarray,
    central_z_slice: int | None = None,
    z_radius: int | None = 5,
) -> np.ndarray:
    """Restrict a 3D volume to a region around a central z-slice.

    Parameters
    ----------
        volume: np.ndarray
            3D numpy array with Z as the first axis.
        central_z_slice: int or None, optional.
            The central z-slice around which to restrict the volume.
            By default None, in which case the central z-slice is the middle slice.
        z_radius: int or None, optional.
            The number of z-slices to include above and below the central z-slice.
            By default 5,
            If it is None, it is auto computed - but this can be problematic for large volumes.
            Hence the default is a fixed value.

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


def _take_random_samples_from_volume(
    volume: np.ndarray,
    num_samples: int | None = 100_000,
) -> None:
    """Take random samples from the volume as a 1D array.

    Parameters
    ----------
        volume: np.ndarray
            The input volume.
        num_samples: int or None, optional.
            The number of samples to take, all data if None.
            Default is 100_000.

    Returns
    -------
        np.ndarray
            The random samples as a 1D array.
    """
    sample_data = volume.flatten()
    num_total_samples = len(sample_data)
    if num_samples is None or (num_samples > num_total_samples):
        return sample_data

    generator = np.random.default_rng(42)
    return generator.choice(
        sample_data,
        num_samples,
        replace=False,
    )


class ContrastLimitCalculator:
    """
    Base class for contrast limit calculators.

    Comes with a default calculation using percentiles.
    This class is designed to allow setting up the calculator with a volume to work on
    and then compute the contrast limits using the compute_contrast_limit method.
    Subclasses can override this method to provide their own implementation.

    An additional feature for tuning contrast limits is hyperparameter optimisation.
    To benefit from this, subclasses should implement the _objective_function and _define_parameter_space methods.

    Attributes
    ----------
    volume: np.ndarray
        The flattened and downsampled volume to calculate the contrast limits from.
    """

    def __init__(
        self,
        volume: np.ndarray,
        z_radius: int | None = None,
        num_samples: int | None = None,
        central_z_slice: int | None = None,
    ):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray
                3D numpy array with Z as the first axis.
            z_radius: int or None, optional.
                The number of z-slices to include above and below the central z-slice.
                By default None.
            num_samples: int or None, optional
                The number of random samples to take from the volume after clipping
                By default None.
                Setting this to None will use all the data.
            central_z_slice: int or None, optional.
                The central z-slice around which to restrict the volume.
                By default None, in which case the central z-slice is the middle slice.
        """
        volume = _restrict_volume_around_central_z_slice(
            volume,
            central_z_slice,
            z_radius,
        )
        self.volume = _take_random_samples_from_volume(volume, num_samples)

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
        return euclidean(
            self._objective_function(params),
            real_limits,
        )

    @abstractmethod
    def compute_contrast_limit(self, *args, **kwargs) -> tuple[float, float]:
        """Calculate the contrast limits"""

    @abstractmethod
    def _objective_function(self, params):
        """Parse params dict to compute contrast limits"""

    @abstractmethod
    def _define_parameter_space(self, parameter_optimizer: ParameterOptimizer):
        """Use parameter_optimizer to define the parameter space for the params dict"""


class PercentileContrastLimitCalculator(ContrastLimitCalculator):

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
        low_value = np.percentile(self.volume, low_percentile)
        high_value = np.percentile(self.volume, high_percentile)

        try:
            return low_value.compute()[0], high_value.compute()[0]
        except AttributeError:
            return low_value, high_value

    def _objective_function(self, params):
        return self.compute_contrast_limit(
            params["low_percentile"],
            params["high_percentile"],
        )

    def _define_parameter_space(self, parameter_optimizer: ParameterOptimizer):
        parameter_optimizer.space_creator(
            {
                "low_percentile": {"type": "randint", "args": [1, 50]},
                "high_percentile": {"type": "randint", "args": [50, 100]},
            },
        )


class GMMContrastLimitCalculator(ContrastLimitCalculator):

    def compute_contrast_limit(
        self,
        low_variance_mult: float = 3.0,
        high_variance_mult: float = 0.5,
        max_components: int = 3,
    ) -> tuple[float, float]:
        """Calculate the contrast limits using Gaussian Mixture Model.

        Parameters
        ----------
        max_components: int, optional.
            The max number of components to use for the GMM.
            By default 5.
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
        covariance_type = "full"
        sample_data = self.volume

        # Find the best number of components - using BIC
        # BIC is a criterion for model selection among a finite set of models
        # The model with the lowest BIC is preferred.
        bics = np.zeros(shape=(max_components, 2))
        for n in range(1, max_components + 1):
            n = int(n)
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                max_iter=100,
                random_state=42,
                init_params="k-means++",
            )
            try:
                gmm.fit(sample_data.reshape(-1, 1))
            except ValueError:
                bics[n - 1] = [np.inf, n]
            else:
                bics[n - 1] = [gmm.bic(sample_data.reshape(-1, 1)), n]
        min_bic_index = np.argmin(bics[:, 0])
        best_n = int(bics[min_bic_index, 1])

        # With less components, we need to be more conservative
        # Hence the standard_deviation multiplier is higher
        std_dev_multi_dict = {
            1: (2.0, 0.5),
            2: (2.2, 0.65),
            3: (3.0, 0.8),
        }

        # Redo the best with more iterations
        gmm_estimator = GaussianMixture(
            n_components=best_n,
            covariance_type=covariance_type,
            max_iter=300,
            random_state=42,
            init_params="k-means++",
        )
        gmm_estimator.fit(sample_data.reshape(-1, 1))

        # Extract the means and variances
        means = gmm_estimator.means_.flatten()
        covariances = gmm_estimator.covariances_  # (n_components, n_features, n_features)
        variances = covariances.flatten()  # n_features is 1, so this is n_components

        # Pick the GMM component which is closest to the mean of the volume
        volume_mean = np.mean(sample_data)
        closest_mean_index = np.argmin(np.abs(means - volume_mean))
        mean_to_use = means[closest_mean_index]
        std_to_use = np.sqrt(variances[closest_mean_index])

        low_variance_mult, high_variance_mult = std_dev_multi_dict[best_n]

        low_limit, high_limit = (
            mean_to_use - low_variance_mult * std_to_use,
            mean_to_use + high_variance_mult * std_to_use,
        )
        # Ensure that the limits are within the range of the volume
        low_limit = float(max(low_limit, np.min(sample_data)))
        high_limit = float(min(high_limit, np.max(sample_data)))
        return low_limit, high_limit

    def _objective_function(self, params):
        return self.compute_contrast_limit(
            params["low_variance_mult"],
            params["high_variance_mult"],
        )

    def _define_parameter_space(self, parameter_optimizer):
        """NOTE: the range here is very small, for real-tuning, it should be larger."""
        parameter_optimizer.space_creator(
            {
                "low_variance_mult": {"type": "uniform", "args": [2.2, 2.21]},
                "high_variance_mult": {"type": "uniform", "args": [0.6, 0.61]},
            },
        )


class CDFContrastLimitCalculator(ContrastLimitCalculator):

    def _automatic_parameter_estimation(self, gradient_threshold=0.3):
        _, _, gradient, _ = self.calculate_cdf(n_bins=512)

        largest_peak = np.argmax(gradient)
        peak_gradient = gradient[largest_peak]
        # Find the start gradient percentage
        # Before the gradient climbs above 20% of the peak gradient
        # Find the median values of the gradient
        start_of_rising = np.where(gradient > gradient_threshold * peak_gradient)[0][0]
        mean_before_rising = np.mean(gradient[:start_of_rising])
        start_gradient_threshold = mean_before_rising / peak_gradient

        # Find the end gradient percentage
        end_of_flattening = np.where(gradient[start_of_rising:] < gradient_threshold * peak_gradient)[0][0]
        mean_after_rising = np.median(gradient[start_of_rising + end_of_flattening :])
        end_gradient_threshold = mean_after_rising / peak_gradient

        start_gradient_threshold = max(0.01, start_gradient_threshold)
        end_gradient_threshold = max(0.01, end_gradient_threshold)

        return start_gradient_threshold, end_gradient_threshold

    def calculate_cdf(self, n_bins=512):
        """Calculate the Cumulative Distribution Function of the volume.

        Parameters
        ----------
        n_bins: int
            The number of bins for the histogram. By default 512.

        Returns
        -------
        cdf: np.ndarray
            The Cumulative Distribution Function.
        bin_edges: np.ndarray
            The bin edges for the histogram used to calculate the CDF.
        gradient: np.ndarray
            The gradient of the CDF, same length as the CDF.
        x: np.ndarray
            The x values for the CDF, calculated from the bin edges.
        """
        min_value = np.min(self.volume)
        max_value = np.max(self.volume)
        hist, bin_edges = np.histogram(self.volume, bins=n_bins, range=[min_value, max_value])
        cdf = np.cumsum(hist) / np.sum(hist)
        try:
            gradient = np.gradient(cdf.compute())
        except AttributeError:
            gradient = np.gradient(cdf)
        x = np.linspace(min_value, max_value, n_bins)
        return cdf, bin_edges, gradient, x

    def compute_contrast_limit(
        self,
        gradient_threshold: float = 0.3,
    ) -> tuple[float, float]:
        """Calculate the contrast limits using the Cumulative Distribution Function.

        Parameters
        ----------
        gradient_threshold: float, optional.
            The threshold multiplier against the peak gradient.
            This is used to estimate the start and end of the contrast limits.
            By default 0.3.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        # Calculate the histogram of the volume
        _, bin_edges, gradient, _ = self.calculate_cdf(n_bins=512)

        # Find the largest peak in the gradient
        largest_peak = np.argmax(gradient)
        peak_gradient_value = gradient[largest_peak]

        # Find the start and end gradient percentages
        start_gradient, end_gradient = self._automatic_parameter_estimation(gradient_threshold)

        # Find where the gradient starts rising and starts flattening after the peak
        start_of_rising = np.where(gradient > start_gradient * peak_gradient_value)[0][0]
        end_of_flattening = np.where(gradient[largest_peak:] < end_gradient * peak_gradient_value)[0][0]
        end_of_flattening += largest_peak
        start_limit = bin_edges[start_of_rising]
        end_limit = bin_edges[end_of_flattening]

        try:
            limits = (start_limit.compute(), end_limit.compute())
        except AttributeError:
            limits = (start_limit, end_limit)

        # Ensure that the limits are within the range of the volume
        return (
            float(max(limits[0], np.min(self.volume))),
            float(min(limits[1], np.max(self.volume))),
        )

    def _objective_function(self, params):
        return self.compute_contrast_limit(params["gradient_treshold"])

    def _define_parameter_space(self, parameter_optimizer):
        parameter_optimizer.space_creator(
            {
                "gradient_treshold": {"type": "uniform", "args": [0.05, 0.6]},
            },
        )


def combined_contrast_limit_plot(
    cdf: list[list[float], list[float]],
    real_limits: tuple[float, float],
    limits_dict: dict[str, tuple[float, float]],
    output_filename: str | Path,
) -> None:
    """Plot the CDF and the calculated limits."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(cdf[0], cdf[1])
    ax.axvline(real_limits[0], color="b")
    ax.axvline(real_limits[1], color="b")

    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color="b", lw=4)]
    colors_dict = {"gmm": "g", "cdf": "y"}
    min_x = real_limits[0]
    max_x = real_limits[1]
    for key, limits in limits_dict.items():
        min_x = min(min_x, limits[0])
        max_x = max(max_x, limits[1])
        color = colors_dict.get(key, "k")
        ax.axvline(limits[0], color=color)
        ax.axvline(limits[1], color=color)
        custom_lines.append(Line2D([0], [0], color=color, lw=4))

    min_x = min_x - 0.1 * (max_x - min_x)
    max_x = max_x + 0.1 * (max_x - min_x)
    ax.set_xlim(min_x, max_x)

    # Produce a legend
    legend = ["Real Limits"]
    for key in limits_dict:
        legend.append(key + " Limits")
    ax.legend(custom_lines, legend)

    fig.savefig(output_filename)
