"""Methods for computing contrast limits for Neuroglancer image layers."""

from abc import abstractmethod
from typing import Optional

import numpy as np


def _restrict_volume_around_central_z_slice(
    volume: "np.ndarray",
    central_z_slice: Optional[int] = None,
    z_radius: int = 5,
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
            By default 5.

    Returns
    -------
        np.ndarray
            3D numpy array. The restricted volume.
    """
    central_z_slice = central_z_slice or volume.shape[0] // 2
    z_min = max(0, central_z_slice - z_radius)
    z_max = min(volume.shape[0], central_z_slice + z_radius + 1)
    return volume[z_min:z_max, :, :]


class ContrastLimitCalculator:

    def __init__(self, volume: Optional["np.ndarray"] = None):
        """Initialize the contrast limit calculator.

        Parameters
        ----------
            volume: np.ndarray or None, optional.
                3D numpy array with Z as the first axis.
                By default None.
        """
        self.volume = volume

    def set_volume_and_z_limits(
        self,
        volume: "np.ndarray",
        central_z_slice: Optional[int] = None,
        z_radius: int = 5,
    ) -> None:
        """Set the volume and z-limits for calculating contrast limits.

        Parameters
        ----------
            volume: np.ndarray
                3D numpy array with Z as the first axis.
            central_z_slice: int or None, optional.
                The central z-slice around which to restrict the volume.
                By default None, in which case the central z-slice is the middle slice.
            z_radius: int, optional.
                The number of z-slices to include above and below the central z-slice.
                By default 5.
        """
        self.volume = _restrict_volume_around_central_z_slice(
            volume,
            central_z_slice,
            z_radius,
        )

    @abstractmethod
    def calculate_contrast_limits_from_percentiles(
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

        return low_value, high_value

    def calcalate_contrast_limits_from_mean_and_rms(
        self,
        multipler: float = 3.0,
    ) -> tuple[float, float]:
        """Calculate the contrast limits from the mean and RMS.

        Parameters
        ----------
            mean: float
                The mean value of the volume.
            rms: float
                The RMS value of the volume.

        Returns
        -------
            tuple[float, float]
                The calculated contrast limits.
        """
        mean_value = np.mean(self.volume)
        rms_value = np.sqrt(np.mean(self.volume**2))
        width = multipler * rms_value

        return mean_value - width, mean_value + width
