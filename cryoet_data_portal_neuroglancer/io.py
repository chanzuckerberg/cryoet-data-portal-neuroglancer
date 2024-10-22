import json
from typing import TYPE_CHECKING, Any

import dask.array as da
import ndjson
import trimesh
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

if TYPE_CHECKING:
    from pathlib import Path


def load_omezarr_data(
    input_filepath: str,
    resolution_level: int = 0,
    persist: bool = True,
) -> da.Array:
    """Load the OME-Zarr data and return a dask array

    Parameters
    ----------
        input_filepath: str
            Path to the OME-Zarr file.
        resolution_level: int, optional
            Resolution level to load.
            By default 0 - the highest resolution.
        persist: bool, optional
            Whether to persist the dask array.
            By default True.

    Returns
    -------
        dask.array.Array
            The loaded OME-Zarr data as a dask array.
    """
    url = parse_url(input_filepath)
    if not url:
        raise ValueError(f"Input file {input_filepath} is not a ZARR file")
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[resolution_level]
    return dask_data.persist() if persist else dask_data


def load_glb_file(glb_file: "Path") -> trimesh.Scene:
    return trimesh.load(glb_file, file_type="glb", force="scene")


def load_oriented_point_data(
    metadata_path: "Path",
    annotations_path: "Path",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load in the metadata (json) and annotations (ndjson) files."""
    with open(metadata_path, mode="r") as f:
        metadata = json.load(f)
    with open(annotations_path, mode="r") as f:
        annotations = ndjson.load(f)
    return metadata, annotations
