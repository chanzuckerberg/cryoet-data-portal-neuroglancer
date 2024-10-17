"""Methods for computing contrast limits for Neuroglancer image layers."""

import logging
from pathlib import Path
from typing import Literal, Optional

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
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
    calculator = GMMContrastLimitCalculator(data) if method == "gmm" else CDFContrastLimitCalculator(data)
    if z_radius is not None:
        if z_radius == "auto":
            z_radius = 15 if method == "gmm" else 5
        calculator.trim_volume_around_central_zslice(z_radius=z_radius)
    if downsampling_ratio is not None:
        total_size = np.prod(data.shape)
        calculator.take_random_samples_from_volume(
            num_samples=int(total_size * downsampling_ratio),
        )
    return calculator.compute_contrast_limit()


def _euclidean_distance(x: tuple[float, float], y: tuple[float, float]) -> float:
    return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def _restrict_volume_around_central_z_slice(
    volume: "np.ndarray",
    central_z_slice: Optional[int] = None,
    z_radius: Optional[int] = 5,
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
        sample_data = self.volume.flatten()
        num_total_samples = len(sample_data)
        if num_samples > num_total_samples:
            return

        generator = np.random.default_rng(0)
        if len(self.volume.flatten()) > num_samples:
            self.volume = generator.choice(
                self.volume.flatten(),
                num_samples,
                replace=False,
            )

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
        return _euclidean_distance(
            self._objective_function(params),
            real_limits,
        )

    def _objective_function(self, params):
        return self.compute_contrast_limit(params["low_percentile"], params["high_percentile"])

    def _define_parameter_space(self, parameter_optimizer: ParameterOptimizer):
        """NOTE: the range here is very small, for real-tuning, it should be larger."""
        parameter_optimizer.space_creator(
            {
                "low_percentile": {"type": "randint", "args": [1, 2]},
                "high_percentile": {"type": "randint", "args": [80, 81]},
            },
        )

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
        sample_data = self.volume.flatten()

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
        self.gmm_estimator = GaussianMixture(
            n_components=best_n,
            covariance_type=covariance_type,
            max_iter=300,
            random_state=42,
            init_params="k-means++",
        )
        self.gmm_estimator.fit(sample_data.reshape(-1, 1))

        # Extract the means and variances
        means = self.gmm_estimator.means_.flatten()
        covariances = self.gmm_estimator.covariances_  # (n_components, n_features, n_features)
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

    def __init__(self, volume: Optional["np.ndarray"] = None):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray or None, optional.
                Input volume for calculating contrast limits.
        """
        super().__init__(volume)
        self.cdf = None

    def automatic_parameter_estimation(self, gradient_threshold=0.3):
        _, _, gradient, _ = self._caculate_cdf(n_bins=512)

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

    def _caculate_cdf(self, n_bins):
        min_value = np.min(self.volume.flatten())
        max_value = np.max(self.volume.flatten())
        hist, bin_edges = np.histogram(self.volume.flatten(), bins=n_bins, range=[min_value, max_value])
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
        cdf, bin_edges, gradient, x = self._caculate_cdf(n_bins=512)

        # Find the largest peak in the gradient
        largest_peak = np.argmax(gradient)
        peak_gradient_value = gradient[largest_peak]

        # Find the start and end gradient percentages
        start_gradient, end_gradient = self.automatic_parameter_estimation(gradient_threshold)

        # Find where the gradient starts rising and starts flattening after the peak
        start_of_rising = np.where(gradient > start_gradient * peak_gradient_value)[0][0]
        end_of_flattening = np.where(gradient[largest_peak:] < end_gradient * peak_gradient_value)[0][0]
        end_of_flattening += largest_peak
        start_limit = bin_edges[start_of_rising]
        end_limit = bin_edges[end_of_flattening]

        self.cdf = [x, cdf]
        try:
            limits = (start_limit.compute(), end_limit.compute())
        except AttributeError:
            limits = (start_limit, end_limit)

        # Ensure that the limits are within the range of the volume
        return (
            float(max(limits[0], np.min(self.volume.flatten()))),
            float(min(limits[1], np.max(self.volume.flatten()))),
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
