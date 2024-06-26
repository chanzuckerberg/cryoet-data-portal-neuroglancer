import numpy as np
import pytest

from cryoet_data_portal_neuroglancer.utils import (
    get_grid_size_from_block_shape,
    number_of_encoding_bits,
    rotate_vector_via_matrix,
    rotate_xyz_via_matrix,
)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 2),
        (16, 4),
        (17, 8),
        (32, 8),
        (2**32, 32),
    ],
)
def test__number_of_encoding_bits(n, expected):
    assert number_of_encoding_bits(n) == expected


def test___number_of_encoding_bits__too_many_values():
    with pytest.raises(ValueError):
        number_of_encoding_bits(2**33)

    with pytest.raises(ValueError):
        number_of_encoding_bits(2**64)


@pytest.mark.parametrize(
    "dshape, bshape, expected",
    [
        ((16, 16, 16), (8, 8, 8), (2, 2, 2)),
        ((17, 17, 17), (8, 8, 8), (3, 3, 3)),
        ((26, 26, 26), (8, 8, 8), (4, 4, 4)),
    ],
)
def test__get_grid_size_from_block_shape(dshape, bshape, expected):
    assert get_grid_size_from_block_shape(dshape, bshape) == expected


def test__rotate_vector_via_matrix():
    # 90 degrees rotation around the z axis
    # Expect to get the X vector to the Y vector
    vec = np.array([1, 0, 0])
    z_rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected_result = np.array([0, 1, 0])
    assert np.allclose(rotate_vector_via_matrix(vec, z_rotation_matrix), expected_result)

    # 90 degrees rotation around the y axis
    # Expect to get the X vector to the -Z vector
    vec = np.array([1, 0, 0])
    y_rotation_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    expected_result = np.array([0, 0, -1])
    assert np.allclose(rotate_vector_via_matrix(vec, y_rotation_matrix), expected_result)

    # 90 degrees rotation around the x axis
    # Expect to get the Y vector to the Z vector
    vec = np.array([0, 1, 0])
    x_rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    expected_result = np.array([0, 0, 1])
    assert np.allclose(rotate_vector_via_matrix(vec, x_rotation_matrix), expected_result)


def test__rotate_xyz_via_matrix():
    # 90 degrees rotation around the z axis
    # Expect to get the X axis to the Y axis, and the Y axis to the -X axis
    z_rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected_result = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(rotate_xyz_via_matrix(z_rotation_matrix), expected_result)
    # 90 degrees rotation around the y axis
    # Expect to get the X axis to the -Z axis, and the Z axis to the X axis
    y_rotation_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    expected_result = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(rotate_xyz_via_matrix(y_rotation_matrix), expected_result)
    # 90 degrees rotation around the x axis
    # Expect to get the Y axis to the Z axis, and the Z axis to the -Y axis
    x_rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    expected_result = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.allclose(rotate_xyz_via_matrix(x_rotation_matrix), expected_result)
