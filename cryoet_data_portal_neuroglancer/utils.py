import logging
from functools import lru_cache
from math import ceil
from typing import TYPE_CHECKING, Iterator

import dask.array as da
import numpy as np
import trimesh

if TYPE_CHECKING:
    import dask.array as da

LOGGER = logging.getLogger(__name__)


def get_scale(
    resolution: tuple[float, float, float] | list[float] | float,
) -> tuple[float, float, float]:
    if not isinstance(resolution, (tuple, list)):
        resolution = [resolution]
    if len(resolution) == 1:
        resolution = (resolution[0],) * 3  # type: ignore
    if len(resolution) != 3:
        raise ValueError("Resolution tuple must have 3 values")
    if any(x <= 0 for x in resolution):
        raise ValueError("Resolution component has to be > 0")
    return resolution  # type: ignore


def pad_block(block: np.ndarray, block_size: tuple[int, int, int]) -> np.ndarray:
    """Pad the block to the given block size with zeros"""
    return np.pad(
        block,
        (
            (0, block_size[0] - block.shape[0]),
            (0, block_size[1] - block.shape[1]),
            (0, block_size[2] - block.shape[2]),
        ),
        # mode='edge'
    )


def iterate_chunks(
    dask_data: da.Array,
) -> Iterator[tuple[da.Array, tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    """Iterate over the chunks in the dask array"""
    chunk_layout = dask_data.chunks

    for zi, z in enumerate(chunk_layout[0]):
        for yi, y in enumerate(chunk_layout[1]):
            for xi, x in enumerate(chunk_layout[2]):
                chunk = dask_data.blocks[zi, yi, xi]

                # Calculate the chunk dimensions
                start = (
                    sum(chunk_layout[0][:zi]),
                    sum(chunk_layout[1][:yi]),
                    sum(chunk_layout[2][:xi]),
                )
                end = (start[0] + z, start[1] + y, start[2] + x)
                dimensions = (start, end)
                yield chunk, dimensions


def get_grid_size_from_block_shape(
    data_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Calculate the grid size from the block shape and data shape

    Both the data shape and block size should be in z, y, x order

    Parameters
    ----------
    data_shape : tuple[int, int, int]
        The shape of the data
    block_shape : tuple[int, int, int]
        The block shape

    Returns
    -------
    tuple[int, int, int]
        The grid size as gz, gy, gx
    """
    gz = ceil(data_shape[0] / block_shape[0])
    gy = ceil(data_shape[1] / block_shape[1])
    gx = ceil(data_shape[2] / block_shape[2])
    return gz, gy, gx


@lru_cache()
def number_of_encoding_bits(nb_values: int) -> int:
    """
    Return the number of bits needed to encode a number of values

    Parameters
    ----------
    nb_values : int
        The number of values that needs to be encoded

    Returns
    -------
    int between (0, 1, 2, 4, 8, 16, 32)
        The number of bits necessary
    """
    for nb_bits in (0, 1, 2, 4, 8, 16, 32):
        if (1 << nb_bits) >= nb_values:
            return nb_bits
    raise ValueError("Too many unique values in block")


def rotate_vector_via_matrix(vec3: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector using a 3x3 rotation matrix

    Parameters
    ----------
    vec3 : np.ndarray
        The 3D vector to rotate
    matrix : np.ndarray
        The 3x3 rotation matrix

    Returns
    -------
    np.ndarray
        The rotated vector
    """
    return np.dot(matrix, vec3)


def rotate_xyz_via_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Rotate the XYZ axes using a 3x3 rotation matrix

    Parameters
    ----------
    matrix : np.ndarray
        The 3x3 rotation matrix

    Returns
    -------
    np.ndarray
        The rotated XYZ axes
    """
    return np.dot(matrix, np.eye(3)).T


def rotate_and_translate_mesh(
    mesh: trimesh.Trimesh,
    scene: trimesh.Scene,
    id_: str | int,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> trimesh.Scene:
    """
    Rotate and translate a mesh using a 3x3 or 4x4 rotation matrix
    and a 3D translation vector

    Parameters
    ----------
    mesh: Trimesh.Mesh
        The mesh to rotate and translate
    scene: Trimesh.Scene
        The scene containing the mesh
    id_: str | int
        The ID of the mesh
    matrix: np.ndarray
        The 3x3 or 4x4 rotation matrix
    translation_vector: np.ndarray
        The 3D translation vector

    Returns
    -------
    Trimesh.Scene
        The scene with the rotated and translated mesh
    """

    def _convert_to_homogenous(rotation_matrix):
        homogenous_matrix = np.eye(4)
        homogenous_matrix[:3, :3] = rotation_matrix
        return homogenous_matrix

    if rotation_matrix.shape == (3, 3):
        transform = _convert_to_homogenous(rotation_matrix)
    elif rotation_matrix.shape == (4, 4):
        transform = rotation_matrix
    else:
        raise ValueError("Rotation matrix must be 3x3 or 4x4")

    transform[:3, 3] = translation_vector

    transformed_mesh = mesh.copy().apply_transform(transform)
    scene.add_geometry(transformed_mesh, node_name=str(id_))

    return scene


def subsample_scene(
    scene: "trimesh.Scene",
    num_elements: int | None = None,
    keys_to_sample: list | None = None,
    at_random: bool = False,
):
    """Subsample the scene to a smaller scene with a given number of elements or keys to sample

    Parameters
    ----------
    scene : trimesh.Scene
        The scene to subsample
    num_elements : int, optional
        The number of elements to sample, by default None
    keys_to_sample : list, optional
        The keys to sample, by default None
    at_random : bool, optional
        Whether to sample at random, by default False
    """
    if (num_elements and keys_to_sample) or (not num_elements and not keys_to_sample):
        raise ValueError("Either num_elements or keys_to_sample should be provided, but not both")
    if num_elements:
        if at_random:
            keys_to_sample = np.random.choice(list(scene.geometry.keys()), num_elements)
        else:  # Take the first num_elements
            keys_to_sample = list(scene.geometry.keys())[:num_elements]
    selected_geometries = {k: v for k, v in scene.geometry.items() if k in keys_to_sample}
    return trimesh.Scene(selected_geometries)


def get_window_limits_from_contrast_limits(
    contrast_limits: tuple[float, float],
    distance_scale: float = 0.1,
) -> tuple[float, float]:
    """
    Create default window limits from contrast limits, 10% padding

    Parameters
    ----------
    contrast_limits : tuple[float, float]
        The contrast limits

    Returns
    -------
    tuple[float, float]
        The window limits
    """
    lower_contrast, higher_contrast = contrast_limits
    # First check if the contrast limits are inverted
    if lower_contrast > higher_contrast:
        lower_contrast, higher_contrast = higher_contrast, lower_contrast

    distance = higher_contrast - lower_contrast
    window_start = lower_contrast - (distance * distance_scale)
    window_end = higher_contrast + (distance * distance_scale)
    return window_start, window_end


def determine_size_of_non_zero_bounding_box(
    data: "da.Array",
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Determine the size of the non-zero bounding box of a volume.

    The input volume is assumed in Z, Y, X order, and the output bounding box
    is in X, Y, Z order for Neuroglancer.

    Parameters
    ----------
    volume : np.ndarray
        The volume to determine the bounding box of

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        The bounding box as a tuple of tuples
    """
    LOGGER.debug("Determining size of non-zero bounding box")
    min_z, max_z = np.inf, 0
    min_y, max_y = np.inf, 0
    min_x, max_x = np.inf, 0
    # Process the data in chunks
    for chunk, _ in iterate_chunks(data):
        non_zero_indices = chunk.nonzero()
        non_zero_indices[0].compute_chunk_sizes()
        if len(non_zero_indices[0]) == 0:
            continue
        min_z = min(min_z, np.min(non_zero_indices[0]))
        max_z = max(max_z, np.max(non_zero_indices[0]))
        min_y = min(min_y, np.min(non_zero_indices[1]))
        max_y = max(max_y, np.max(non_zero_indices[1]))
        min_x = min(min_x, np.min(non_zero_indices[2]))
        max_x = max(max_x, np.max(non_zero_indices[2]))

    if min_z == np.inf:
        min_z = min_y = min_x = 0
    return max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1


def determine_mesh_shape_from_lods(lods: list[trimesh.Trimesh]):
    mesh_starts = [np.min(lod.vertices, axis=0) for lod in lods]
    mesh_ends = [np.max(lod.vertices, axis=0) for lod in lods]
    LOGGER.debug(
        "LOD mesh origin points %s and end points %s",
        mesh_starts,
        mesh_ends,
    )
    grid_origin = np.floor(np.min(mesh_starts, axis=0))
    grid_end = np.ceil(np.max(mesh_ends, axis=0))
    mesh_shape = (grid_end - grid_origin).astype(int)
    return grid_origin, mesh_shape
