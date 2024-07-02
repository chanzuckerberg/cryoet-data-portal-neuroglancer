from typing import TYPE_CHECKING

import dask.array as da
import trimesh
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

if TYPE_CHECKING:
    from pathlib import Path


def load_omezarr_data(input_filepath: str) -> da.Array:
    """Load the OME-Zarr data and return a dask array"""
    url = parse_url(input_filepath)
    if not url:
        raise ValueError(f"Input file {input_filepath} is not a ZARR file")
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[0]
    return dask_data.persist()


def load_glb_file(glb_file: "Path") -> trimesh.Trimesh:
    return trimesh.load(glb_file, file_type="glb", force="mesh")
