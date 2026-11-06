from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from zarr.storage import LocalStore

AXIS_TYPES = {"t": "time", "c": "channel"}


def write_ome_zarr(
    path: Path,
    data: np.ndarray,
    *,
    version: Literal["0.4", "0.5"] = "0.5",
    axes: list[str] | None = None,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    levels: int = 1,
    voxel_size: float = 1.0,
    attributes: dict[str, Any] | None = None,
    label_values: list[int] | None = None,
    label_location: Literal["ome", "multiscales-metadata"] = "ome",
) -> Path:
    """Write a small OME-Zarr image: Zarr v2 / OME-NGFF 0.4 or Zarr v3 / OME-NGFF 0.5 (optionally sharded).

    Levels are made by striding the last three axes by 2. ``attributes`` go on every level array.
    """
    zarr_format = 3 if version == "0.5" else 2
    axes = axes or ["z", "y", "x"]
    root = zarr.open_group(LocalStore(path), mode="w", zarr_format=zarr_format)
    datasets = []
    level = data
    for k in range(levels):
        kwargs: dict[str, Any] = {"chunks": chunks or level.shape, "fill_value": 0, "attributes": attributes or {}}
        if zarr_format == 3:
            kwargs.update(shards=shards, dimension_names=axes)
        else:
            kwargs.update(chunk_key_encoding={"name": "v2", "separator": "/"})
        arr = root.create_array(str(k), shape=level.shape, dtype=level.dtype, **kwargs)
        arr[...] = level
        scale = [1.0 if a in AXIS_TYPES else voxel_size * 2**k for a in axes]
        datasets.append({"path": str(k), "coordinateTransformations": [{"type": "scale", "scale": scale}]})
        level = level[..., ::2, ::2, ::2]
    multiscale: dict[str, Any] = {
        "axes": [
            {"name": a, "type": AXIS_TYPES.get(a, "space"), **({} if a in AXIS_TYPES else {"unit": "angstrom"})}
            for a in axes
        ],
        "datasets": datasets,
        "name": "test",
    }
    image_label = None if label_values is None else {"colors": [{"label-value": v} for v in label_values]}
    if image_label is not None and label_location == "multiscales-metadata":
        multiscale["metadata"] = {"image-label": {"version": "0.4", **image_label}}
        image_label = None
    if version == "0.5":
        ome: dict[str, Any] = {"version": "0.5", "multiscales": [multiscale]}
        if image_label is not None:
            ome["image-label"] = image_label
        root.attrs["ome"] = ome
    else:
        multiscale["version"] = "0.4"
        root.attrs["multiscales"] = [multiscale]
        if image_label is not None:
            root.attrs["image-label"] = {"version": "0.4", **image_label}
    return path
