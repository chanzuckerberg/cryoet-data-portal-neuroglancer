from functools import lru_cache
from math import ceil
from typing import TYPE_CHECKING, Iterator

import dask.array as da
import numpy as np

if TYPE_CHECKING:
    from trimesh import Trimesh


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
    mesh: "Trimesh.Mesh",
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    should_copy: bool = True,
) -> "Trimesh.Mesh":
    """
    Rotate and translate a mesh using a 3x3 or 4x4 rotation matrix
    and a 3D translation vector

    Parameters
    ----------
    mesh : Trimesh.Mesh
        The mesh to rotate and translate
    matrix : np.ndarray
        The 3x3 or 4x4 rotation matrix
    translation_vector : np.ndarray
        The 3D translation vector
    should_copy : bool
        If True, the mesh will be copied before transforming

    Returns
    -------
    Trimesh.Mesh
        The rotated and translated mesh
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

    if should_copy:
        return mesh.copy().apply_transform(transform)
    return mesh.apply_transform(rotation_matrix)
