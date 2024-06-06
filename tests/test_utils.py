import pytest

from cryoet_data_portal_neuroglancer.utils import (
    get_grid_size_from_block_shape,
    get_window_limits_from_contrast_limits,
    number_of_encoding_bits,
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


@pytest.mark.parametrize(
    "contrast_limits, window_limits",
    [
        ((0.0, 1.0), (-0.1, 1.1)),
        ((-5.0, 5.0), (-6.0, 6.0)),
        ((20, 10), (9, 21)),
        ((100, -100), (-120, 120)),
    ],
)
def test_create_default_window_limits_from_contrast_limits(contrast_limits, window_limits):
    assert get_window_limits_from_contrast_limits(contrast_limits) == window_limits
    assert contrast_limits == contrast_limits
