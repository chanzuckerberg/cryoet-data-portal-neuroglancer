import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
from tqdm import tqdm


def _vr_worker(
    pixel_indices_chunk,
    shm_name_volume,
    volume_shape,
    contrast_limits,
    exponential_gain,
    depth_samples,
    shm_name_image,
    image_shape,
):
    existing_shm_volume = shared_memory.SharedMemory(name=shm_name_volume)
    volume = np.ndarray(volume_shape, dtype=np.float32, buffer=existing_shm_volume.buf)

    existing_shm_image = shared_memory.SharedMemory(name=shm_name_image)
    volume_rendered_image = np.ndarray(image_shape, dtype=np.float32, buffer=existing_shm_image.buf)

    for y, x in pixel_indices_chunk:
        color, opacity = _direct_composite_along_ray(
            (x, y),
            depth_samples,
            volume,
            contrast_limits,
            exponential_gain,
        )
        volume_rendered_image[y, x] = [*color, opacity]

    existing_shm_volume.close()
    existing_shm_image.close()


def volume_render(
    volume: "np.ndarray",
    contrast_limits: tuple[float, float],
    exponential_gain: float = 0.0,
    depth_samples: int = 64,
    num_workers: Optional[int] = None,
):
    if num_workers is None:
        num_workers = mp.cpu_count()

    volume_shape = volume.shape
    volume_rendered_image_shape = (volume.shape[1], volume.shape[2], 4)

    # Create shared memory for volume
    shm_volume = shared_memory.SharedMemory(create=True, size=volume.nbytes)
    shared_volume = np.ndarray(volume_shape, dtype=np.float32, buffer=shm_volume.buf)
    shared_volume[:] = volume[:]

    # Create shared memory for the rendered image
    shm_image = shared_memory.SharedMemory(
        create=True, size=np.prod(volume_rendered_image_shape) * np.float32().itemsize
    )

    try:
        pixel_indices = list(np.ndindex(volume.shape[1], volume.shape[2]))
        chunk_size = len(pixel_indices) // num_workers

        chunks = [pixel_indices[i : i + chunk_size] for i in range(0, len(pixel_indices), chunk_size)]

        processes = [
            mp.Process(
                target=_vr_worker,
                args=(
                    chunk,
                    shm_volume.name,
                    volume_shape,
                    contrast_limits,
                    exponential_gain,
                    depth_samples,
                    shm_image.name,
                    volume_rendered_image_shape,
                ),
            )
            for chunk in chunks
        ]

        for p in processes:
            p.start()

        for p in tqdm(processes, desc="Processing volume rendering in chunks"):
            p.join()
    except Exception as e:
        print(f"Error processing volume rendering: {e}")
    # Clean up shared memory
    finally:
        shm_volume.close()
        shm_volume.unlink()

    return shm_image, volume_rendered_image_shape


def _inverse_lerp(start_value: float, end_value: float, value: float) -> float:
    return (value - start_value) / (end_value - start_value)


def _lerp(start_value: float, end_value: float, t: float) -> float:
    return start_value + t * (end_value - start_value)


def _order_independent_transparency_wieght(alpha: float, depth: float) -> float:
    a = min(1.0, alpha) * 8.0 + 0.01
    b = -depth * 0.95 + 1.0
    return a * a * a * b * b * b


def _compute_depth_from_sample_index(sample_index: int, depth_samples: int) -> float:
    # Make the range small to reduce impact
    start = -0.1
    end = 0.1

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
    return np.clip(final_color * final_opacity, 0.0, 1.0), np.clip(final_opacity**2, 0.0, 1.0)
