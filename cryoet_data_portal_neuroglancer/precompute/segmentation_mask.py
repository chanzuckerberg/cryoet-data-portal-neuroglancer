import functools
import json
import logging
import shutil
import struct
from pathlib import Path
from typing import Any, Iterator

import dask.array as da
import igneous.task_creation as tc
import neuroglancer
import numpy as np
from taskqueue import LocalTaskQueue
from tqdm import tqdm

import cryoet_data_portal_neuroglancer.igneous_patch as igneous_patch
from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.models.chunk import Chunk
from cryoet_data_portal_neuroglancer.precompute.mesh import determine_chunk_size_for_lod
from cryoet_data_portal_neuroglancer.utils import (
    determine_size_of_non_zero_bounding_box,
    get_grid_size_from_block_shape,
    iterate_chunks,
    number_of_encoding_bits,
    pad_block,
)

# Patch igneous
igneous_patch.patch()

LOGGER = logging.getLogger(__name__)


def _get_buffer_position(buffer: bytearray) -> int:
    """Return the current position in the buffer"""
    assert len(buffer) % 4 == 0, "Buffer length must be a multiple of 4"
    return len(buffer) // 4


def _create_block_header(
    buffer: bytearray,
    lookup_table_offset: int,
    encoded_bits: int,
    encoded_values_offset: int,
    block_offset: int,
) -> None:
    """
    Create a block header (64-bit)

    First 24 bits are the lookup table offset (little endian)
    Next 8 bits are the number of bits used to encode the values
    Last 32 bits are the offset to the encoded values (little endian)
    All values are unsigned integers

    Parameters
    ----------
    buffer : bytearray
        The buffer to write the block header to
    lookup_table_offset : int
        The offset in the buffer to the lookup table for this block
    encoded_bits : int
        The number of bits used to encode the values
    encoded_values_offset : int
        The offset in the buffer to the encoded values for this block
    block_offset : int
        The offset in the buffer to the block header
    """
    struct.pack_into(
        "<II",
        buffer,
        block_offset,
        lookup_table_offset | (encoded_bits << 24),
        encoded_values_offset,
    )


def _create_lookup_table(
    buffer: bytearray,
    stored_lookup_tables: dict[bytes, tuple[int, int]],
    unique_values: np.ndarray,
) -> tuple[int, int]:
    """
    Create a lookup table for the given values

    Parameters
    ----------
    buffer : bytearray
        The buffer to write the lookup table to
    stored_lookup_tables : dict[bytes, int]
        A dictionary mapping values to their offset in the buffer
    unique_values : np.ndarray
        The values to write to the buffer
        Must be uint32 or uint64

    Returns
    -------
    lookup_table_offset : int
        The offset in the buffer to the lookup table for the given values
    encoded_bits : int
        The number of bits used to encode the values
    """
    unique_values = unique_values.astype(np.uint32)
    values_in_bytes = unique_values.tobytes()
    if values_in_bytes not in stored_lookup_tables:
        lookup_table_offset = _get_buffer_position(buffer)
        encoded_bits = number_of_encoding_bits(len(unique_values))
        stored_lookup_tables[values_in_bytes] = (
            lookup_table_offset,
            encoded_bits,
        )
        buffer += values_in_bytes
    else:
        lookup_table_offset, encoded_bits = stored_lookup_tables[values_in_bytes]
    return lookup_table_offset, encoded_bits


def _pack_encoded_values(values: np.ndarray, bits: int) -> bytes:
    """
    Pack the encoded values into 32bit unsigned integers

    To view the packed values as a numpy array, use the following:
    np.frombuffer(packed_values, dtype=np.uint32).view(f"u{encoded_bits}")

    Parameters
    ----------
    values : np.ndarray
        The values to encode
    bits : int
        The number of bits used to encode the values

    Returns
    -------
    packed_values : bytes
        The packed values

    Details
    -------

    Values are packed in a little endian 32bits, from LSB to MSB.
    Consequently, the first value of the array will be stored first from the LSB
    then, shifted to the left from the nb of bits necessary to encode the value,
    the next value from the array is considered.
    Each small encoded value are reduced in a huge 32bits using a simple bits | operator

    Here is an example for an array [1, 0, 2, 2, 1].
    There is only 3 different values here, so we need to encode each value on 2bits
    The result would then be:
    values    1   2   2   0   1
    encoded  01  10  10  00  01
    result 0b110100001 packed in a unsigned 32bit little endian
    """
    if bits == 0:
        return bytes()
    assert 32 % bits == 0
    assert np.array_equal(values, values & ((1 << bits) - 1))
    values_per_32bit = 32 // bits
    padded_values = np.pad(
        values.astype("<I", casting="unsafe"),
        [(0, -len(values) % values_per_32bit)],
        mode="constant",
        constant_values=0,
    )
    assert len(padded_values) % values_per_32bit == 0
    packed_values: np.ndarray = functools.reduce(
        np.bitwise_or,
        (padded_values[shift::values_per_32bit] << (shift * bits) for shift in range(values_per_32bit)),
    )
    return packed_values.tobytes()
    # packed_values: int = functools.reduce(
    #     operator.or_,
    #     (value << (shift * bits) for shift, value in enumerate(padded_values)),
    # )
    # return struct.pack("<I", packed_values)


def _create_encoded_values(buffer: bytearray, positions: np.ndarray, encoded_bits: int) -> int:
    """Create the encoded values for the given values

    Parameters
    ----------
    buffer: bytearray
        The buffer to write the encoded values to
    positions: np.ndarray
        The values to encode (positions in the lookup table)
    encoded_bits: int
        The number of bits used to encode the values

    Returns
    -------
    encoded_values_offset: int
        The offset in the buffer to the encoded values
    """
    encoded_values_offset = _get_buffer_position(buffer)
    buffer += _pack_encoded_values(positions, encoded_bits)
    return encoded_values_offset


def _create_file_chunk_header(number_channels: int = 1) -> bytearray:
    buf = bytearray(4 * number_channels)
    for offset in range(number_channels):
        struct.pack_into("<I", buf, offset * 4, len(buf) // 4)
    return buf


def create_segmentation_chunk(
    data: np.ndarray,
    dimensions: tuple[tuple[int, int, int], tuple[int, int, int]],
    block_size: tuple[int, int, int] = (8, 8, 8),
    convert_non_zero_to: int | None = 0,
) -> Chunk:
    """Convert data in a dask array to a neuroglancer segmentation chunk"""
    bz, by, bx = block_size
    if len(data.shape) != 3:
        raise ValueError("Data must be 3-dimensional")
    if convert_non_zero_to:
        data[data > 0] = convert_non_zero_to
        data[data < 0] = 0
    gz, gy, gx = get_grid_size_from_block_shape(data.shape, block_size)  # type: ignore
    stored_lookup_tables: dict[bytes, tuple[int, int]] = {}

    # big enough to hold the 64-bit starting block headers
    buffer = bytearray(gx * gy * gz * 8)

    # data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
    for z, y, x in np.ndindex((gz, gy, gx)):
        block = data[z * bz : (z + 1) * bz, y * by : (y + 1) * by, x * bx : (x + 1) * bx]
        if block.shape != block_size:
            block = pad_block(block, block_size)
        unique_values, encoded_values = np.unique(block, return_inverse=True)

        lookup_table_offset, encoded_bits = _create_lookup_table(buffer, stored_lookup_tables, unique_values)
        encoded_values_offset = _create_encoded_values(buffer, encoded_values, encoded_bits)
        block_offset = 8 * (x + gx * (y + gy * z))
        _create_block_header(
            buffer,
            lookup_table_offset,
            encoded_bits,
            encoded_values_offset,
            block_offset,
        )

    return Chunk(_create_file_chunk_header() + buffer, dimensions)


def _create_metadata(
    chunk_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
    data_size: tuple[int, int, int],
    data_directory: str,
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mesh_directory: str | None = None,
) -> dict[str, Any]:
    """Create the metadata for the segmentation"""
    metadata = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [chunk_size],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": block_size,
                "resolution": resolution,
                "key": data_directory,
                "size": data_size[::-1],  # reverse the data size to pass from Z-Y-X to X-Y-Z
            },
        ],
        "type": "segmentation",
    }
    if mesh_directory:
        metadata["mesh"] = mesh_directory
    return metadata


def create_segmentation(
    dask_data: da.Array,
    block_size: tuple[int, int, int],
    convert_non_zero_to: int | None = 0,
) -> Iterator[Chunk]:
    """Yield the neuroglancer segmentation format chunks"""
    to_iterate = iterate_chunks(dask_data)
    num_iters = np.prod(dask_data.numblocks)
    for chunk, dimensions in tqdm(to_iterate, desc="Processing chunks", total=num_iters):
        yield create_segmentation_chunk(
            chunk.compute(),
            dimensions,
            block_size,
            convert_non_zero_to=convert_non_zero_to,
        )


def create_mesh(
    dask_data: da.Array,
    output_path: Path,
    mesh_directory: str,
    resolution: tuple[float, float, float],
) -> None:
    """Create the mesh for the given volume if a mesh directory is provided"""
    mesh = np.dstack([np.array(dask_data).astype(np.uint8)])
    transposed_mesh = np.transpose(mesh, (2, 1, 0))

    ids = [int(i) for i in np.unique(transposed_mesh[:])]
    coordinate_space = neuroglancer.CoordinateSpace(
        names=("x", "y", "z"),
        units=("m",) * 3,
        scales=resolution,
    )
    vol = neuroglancer.LocalVolume(data=transposed_mesh, dimensions=coordinate_space)

    mesh_path = output_path / mesh_directory
    mesh_path.mkdir(exist_ok=True, parents=True)
    json_descriptor = '{{"fragments": ["mesh.{}.{}"]}}'
    for id in ids[1:]:
        mesh_data = vol.get_object_mesh(id)
        with open(str(mesh_path / ".".join(("mesh", str(id), str(id)))), "wb") as mesh_file:
            mesh_file.write(mesh_data)
        with open(str(mesh_path / "".join((str(id), ":0"))), "w") as frag_file:
            frag_file.write(json_descriptor.format(id, id))
    print(f"Wrote segmentation mesh to {mesh_path}")


def generate_mesh(
    precomputed_segmentation_path: Path,
    mesh_directory: str,
    max_lod: int,
    mesh_shape: tuple[int, int, int] | np.ndarray,
    min_mesh_chunk_dim: int = 16,
) -> None:
    """Generates the meshes for a segmentation stored as a precomputed Neuroglancer format.

    Parameters
    ----------
    precomputed_segmentation_path: Path
        The path towards the segmentation stored as a precomputed Neuroglancer format
    mesh_directory: str
        The name of the directory that will receive the mesh information
    max_lod: int
        The maximal lod that needs to be generated, starting from 0
        This may not be achieved if the mesh is too small
    mesh_shape: tuple[int, int, int] | np.ndarray
        The shape of the mesh - used in calculating the chunk size for LOD generation
    min_mesh_chunk_dim: int
        The minimal dimension of a chunk.
    """
    tq = LocalTaskQueue()

    path = f"precomputed://file://{precomputed_segmentation_path}"
    tasks = tc.create_meshing_tasks(path, mip=0, mesh_dir=mesh_directory, sharded=True)
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(path, mesh_dir=mesh_directory, magnitude=3)
    tq.insert(tasks)
    tq.execute()
    min_chunk_size = determine_chunk_size_for_lod(
        mesh_shape,
        max_lod,
        min_mesh_chunk_dim,
    )
    tasks = tc.create_sharded_multires_mesh_tasks(
        path,
        mesh_dir=mesh_directory,
        num_lod=max_lod,
        min_chunk_size=min_chunk_size,
    )
    tq.insert(tasks)
    tq.execute()


def write_metadata(metadata: dict[str, Any], output_directory: Path) -> None:
    """Write the segmentation to the given directory"""
    metadata_path = output_directory / "info"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def encode_segmentation(
    filename: str,
    output_path: Path | str,
    resolution: tuple[float, float, float],
    block_size: tuple[int, int, int] = (64, 64, 64),
    data_directory: str = "data",
    delete_existing: bool = False,
    convert_non_zero_to: int | None = 0,
    include_mesh: bool = False,
    mesh_directory: str = "mesh",
    max_lod: int = 2,
    min_mesh_chunk_dim: int = 16,
) -> None:
    """Convert the given OME-Zarr file to neuroglancer segmentation format with the given block size

    Parameters
    ----------
    filename : str
        The path to the OME-Zarr file
    output_path : Path | str
        The path to the output directory
    resolution : tuple[float, float, float]
        The resolution of the data in nm
    block_size : tuple[int, int, int], optional
        The size of the blocks to use, by default (64, 64, 64)
        This determines the size of the chunks in the precomputed format
        output
    data_directory : str, optional
        The name of the data directory, by default "data"
        This is the directory that will contain the segmentation data
    delete_existing : bool, optional
        Whether to delete the existing output directory, by default False
        If False and the output directory exists, the function will
        return without doing anything
    convert_non_zero_to : int | None, optional
        The value to convert non-zero values to, by default 0, which
        will leave non-zero values as they are. If None, non-zero
        values will be left as they are also. This is useful for
        representing multiple objects in the same segmentation
    include_mesh : bool, optional
        Whether to include mesh information in the output, by default False
    mesh_directory : str, optional
        The name of the mesh directory, by default "mesh"
        This is the directory that will contain the mesh data
    max_lod : int, optional
        The maximum level of detail to generate meshes for, by default 2
        This will produce a high resolution mesh at LOD 0, medium resolution
        at LOD 1, and low resolution at LOD 2
    min_mesh_chunk_dim : int, optional
        The minimum dimension of a mesh chunk, by default 16
        This is used in determining the chunk size for LOD generation
        If the mesh is small, it means that we may not be able to achieve
        the desired LOD unless we process the data in very small chunks.
        Processing in very small chunks is inefficient, and error prone,
        so we set a minimum chunk size to avoid this.
        If you want to force the generation of LODs, you can set this to 1.

    """
    LOGGER.info("Converting %s to neuroglancer compressed segmentation format", filename)
    output_path = Path(output_path)

    dask_data = load_omezarr_data(filename)
    if delete_existing and output_path.exists():
        contents = list(output_path.iterdir())
        content_names = {c.name for c in contents}
        if content_names and content_names.difference({"data", "info", "mesh", "provenance"}):
            raise FileExistsError(
                f"Output directory {output_path!s} exists and contains non-conversion related files. {content_names}",
            )
        else:
            LOGGER.info(
                "The output directory %s exists from a previous run, deleting before starting conversion",
                output_path,
            )
            shutil.rmtree(output_path)
    elif not delete_existing and output_path.exists():
        LOGGER.warning("The output directory %s already exists, skipping", output_path)
        return
    output_path.mkdir(parents=True, exist_ok=True)

    for c in create_segmentation(dask_data, block_size, convert_non_zero_to=convert_non_zero_to):
        c.write_to_directory(output_path / data_directory)

    if len(dask_data.chunksize) != 3:
        raise ValueError(f"Expected 3 chunk dimensions, got {len(dask_data.chunksize)}")

    metadata = _create_metadata(
        dask_data.chunksize,  # type: ignore
        block_size,
        dask_data.shape,
        data_directory,
        resolution,
        mesh_directory=mesh_directory if include_mesh else None,
    )
    write_metadata(metadata, output_path)

    if include_mesh:
        LOGGER.info("Converting %s to neuroglancer mesh format", filename)
        mesh_shape = determine_size_of_non_zero_bounding_box(dask_data.compute())
        generate_mesh(
            output_path,
            mesh_directory,
            max_lod=max_lod,
            mesh_shape=mesh_shape,
            min_mesh_chunk_dim=min_mesh_chunk_dim,
        )

    LOGGER.info("Wrote segmentation to %s", output_path)
