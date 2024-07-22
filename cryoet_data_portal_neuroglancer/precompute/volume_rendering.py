import numpy as np
from tqdm import tqdm


def volume_render(
    volume: "np.ndarray",
    contrast_limits: tuple[float, float],
    exponential_gain: float = 0.0,
    depth_samples: int = 64,
):
    """Volume rendering of a 3D numpy array.

    Parameters
    ----------
        volume: np.ndarray
            3D numpy array to render.
        contrast_limits: tuple[float, float]
            Contrast limits for the volume.
        exponential_gain: float, optional
            Exponential gain for the volume rendering.
            By default 0.0.
        depth_samples: int, optional
            Number of depth samples for the volume rendering.
            By default 64.

    Returns
    -------
        np.ndarray
            Volume rendered image.
    """
    volume_rendered_image = np.zeros(
        (volume.shape[1], volume.shape[2], 4),
        dtype=np.float32,
    )
    for y, x in tqdm(
        np.ndindex(volume.shape[1], volume.shape[2]),
        total=volume.shape[1] * volume.shape[2],
        desc="Volume Rendering on ray",
    ):
        color, opacity = _direct_composite_along_ray(
            (x, y),
            depth_samples,
            volume,
            contrast_limits,
            exponential_gain,
        )
        volume_rendered_image[y, x] = [*color, opacity]

    return volume_rendered_image


def _inverse_lerp(start_value: float, end_value: float, value: float) -> float:
    return (value - start_value) / (end_value - start_value)


def _lerp(start_value: float, end_value: float, t: float) -> float:
    return start_value + t * (end_value - start_value)


def _order_independent_transparency_wieght(alpha: float, depth: float) -> float:
    a = min(1.0, alpha) * 8.0 + 0.01
    b = -depth * 0.95 + 1.0
    return a * a * a * b * b * b


def _compute_depth_from_sample_index(sample_index: int, depth_samples: int) -> float:
    # -1.0 to 1.0, with 0.0 at the center and 1.0 at the end
    start = -0.8  # Remove the first 10% depth
    end = 0.2  # Remove the last 40% of the depth

    return _lerp(start, end, (sample_index + 0.5) / depth_samples)


def _direct_composite_along_ray(
    xy_coordinate: tuple[int, int],
    depth_samples: int,
    volume: "np.ndarray",
    contrast_limits: tuple[float, float],
    exponential_gain: float = 0.0,
):
    actual_gain = np.exp(exponential_gain)
    row_of_data = volume[:, xy_coordinate[1], xy_coordinate[0]]
    accumulated_color_and_opacity = np.zeros(4)
    accumulated_revealage = 1.0
    # Neuroglancer usually throws away a number of depth samples due to the near and far clipping planes
    actual_depth_samples = depth_samples // 2
    for i in range(1, actual_depth_samples + 1):
        step = ((i - 0.5) * volume.shape[0]) / depth_samples
        closest_z = int(np.floor(step))
        raw_data_value = row_of_data[closest_z]
        normalized_data_value = _inverse_lerp(*contrast_limits, raw_data_value)
        corrected_alpha = np.clip(normalized_data_value * actual_gain, 0.0, 1.0)
        weighted_alpha = corrected_alpha * _order_independent_transparency_wieght(
            corrected_alpha,
            _compute_depth_from_sample_index(i, depth_samples),
        )
        color = normalized_data_value * weighted_alpha
        accumulated_color_and_opacity += [color, color, color, weighted_alpha]
        accumulated_revealage *= 1.0 - corrected_alpha

    accumulated_color = accumulated_color_and_opacity[:3]
    accumulated_opacity = accumulated_color_and_opacity[3]
    final_color = accumulated_color / accumulated_opacity
    final_opacity = 1.0 - accumulated_revealage
    return final_color * final_opacity, final_opacity**2
