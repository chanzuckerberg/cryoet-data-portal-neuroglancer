"""Methods for computing contrast limits for Neuroglancer image layers."""

from abc import abstractmethod
from typing import Optional

import numpy as np
from scipy.signal import find_peaks


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
            # TODO create fallback instead
            raise ValueError("Not enough low points found")
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

    @abstractmethod
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
        low_value = np.percentile(self.volume, low_percentile)
        high_value = np.percentile(self.volume, high_percentile)

        return low_value, high_value

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
