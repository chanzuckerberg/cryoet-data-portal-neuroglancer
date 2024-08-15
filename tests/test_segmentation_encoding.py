import struct
from ctypes import LittleEndianStructure, c_uint64

import numpy as np
import pytest

from cryoet_data_portal_neuroglancer.models.chunk import Chunk
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import (
    _create_block_header,
    _create_encoded_values,
    _create_file_chunk_header,
    _create_lookup_table,
    _get_buffer_position,
    _pack_encoded_values,
    create_segmentation_chunk,
)


# Used for decoding the header
class BlockHeader(LittleEndianStructure):
    _fields_ = [
        ("lookup_table_offset", c_uint64, 24),
        ("encoded_bits", c_uint64, 8),
        ("encoded_values_offset", c_uint64, 32),
    ]


def test__get_buffer_position():
    assert _get_buffer_position(bytearray(8)) == 2
    assert _get_buffer_position(bytearray(16)) == 4

    with pytest.raises(AssertionError):
        _get_buffer_position(bytearray(5))


def test__create_block_header__without_offset():
    buffer = bytearray(64 // 8)
    _create_block_header(
        buffer,
        lookup_table_offset=0xBEEF00,
        encoded_bits=0xAB,
        encoded_values_offset=0xDEADBEEF,
        block_offset=0,
    )
    result = BlockHeader.from_buffer(buffer, 0)

    assert result.lookup_table_offset == 0xBEEF00
    assert result.encoded_bits == 0xAB
    assert result.encoded_values_offset == 0xDEADBEEF


def test__create_block_header__with_offset():
    offset = 0x4
    buffer = bytearray(64 // 8 + offset)
    _create_block_header(
        buffer,
        lookup_table_offset=0xBEEF00,
        encoded_bits=0xAB,
        encoded_values_offset=0xDEADBEEF,
        block_offset=offset,
    )
    result = BlockHeader.from_buffer(buffer, offset)

    assert result.lookup_table_offset == 0xBEEF00
    assert result.encoded_bits == 0xAB
    assert result.encoded_values_offset == 0xDEADBEEF


## Need to go back again there
def test__create_lookup_table__fill_global_lut():
    buffer = bytearray.fromhex("DEADBEEF")
    global_lut = {}

    lut_offset, nb_bits = _create_lookup_table(buffer, global_lut, np.array([1, 0, 3]))

    assert nb_bits == 2
    assert len(global_lut) == 1
    assert len(buffer) == 16  # ?
    assert lut_offset == 1  # ?


@pytest.mark.parametrize(
    "array, nb_bits, expected",
    [
        ([1, 0, 2], 2, 0b10_00_01),
        ([1, 0, 2, 3, 4], 4, 0b0100_0011_0010_0000_0001),
    ],
)
def test__pack_encoded_values(array, nb_bits, expected):
    encoded = _pack_encoded_values(np.array(array), nb_bits)
    assert encoded == struct.pack("<I", expected)


def test__pack_encoded_values__needs_32multiple():
    array = [1, 0, 2, 3, 4]
    nb_bits = 3
    with pytest.raises(AssertionError):
        _pack_encoded_values(np.array(array), nb_bits)


def test__create_encoded_values():
    buffer = bytearray()  # will start in 0
    offset = _create_encoded_values(buffer, np.array([1, 0, 2]), 2)
    assert offset == 0
    assert buffer == struct.pack("<I", 0b100001)

    buffer = bytearray(8)  # will start in 2
    offset = _create_encoded_values(buffer, np.array([1, 0, 2]), 2)
    assert offset == 2
    assert buffer == struct.pack("<QI", 0, 0b100001)  # we need to pad manually for the test


def test__create_file_chunck_header():
    buffer = _create_file_chunk_header()
    assert buffer == struct.pack("<I", 1)


def test__create_segmentation_chunk():
    # We take a small 8x8 cube
    array = [
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
    ]
    real_data = np.array(array)
    chunk: Chunk = create_segmentation_chunk(real_data, dimensions=((0, 0, 0), (8, 8, 4)), block_size=(8, 8, 4))

    assert chunk.dimensions == ((0, 0, 0), (8, 8, 4))
    byte_array = chunk.buffer
    data_start_offset = 20 # header of 8 bytes + 12 bytes of info
    data = np.frombuffer(byte_array, dtype=np.uint32, offset=data_start_offset)
    # The data is symmetric, so each 32-bit integer should be the same
    assert len(data) == 8 # 8 * 8 * 4 / 32
    assert np.all(np.diff(data) == 0)

    # If we chunk in larger blocks, it should still work with padding
    chunk: Chunk = create_segmentation_chunk(real_data, dimensions=((0, 0, 0), (8, 8, 4)), block_size=(8, 8, 8))

    assert chunk.dimensions == ((0, 0, 0), (8, 8, 4))
    byte_array = chunk.buffer
    data_start_offset = 20
    data = np.frombuffer(byte_array, dtype=np.uint32, offset=data_start_offset)
    # The data is symmetric, so each 32-bit integer should be the same
    assert len(data) == 16 # 8 * 8 * 8 / 32
    assert np.all(np.diff(data) == 0)

    # With smaller blocks, there should be more of them
    chunk: Chunk = create_segmentation_chunk(real_data, dimensions=((0, 0, 0), (8, 8, 4)), block_size=(4, 4, 4))

    assert chunk.dimensions == ((0, 0, 0), (8, 8, 4))
    byte_array = chunk.buffer
    data_start_offset = 20
    data = np.frombuffer(byte_array, dtype=np.uint32)

    # In this case, there should be four block headers (32 bits) - followed by the 3 32-bit info bytes, and then all the data is the same in one block of 8 32-bit integers
    assert len(data) == 8 + 3 + 8
    assert np.all(np.diff(data[11:]) == 0)
