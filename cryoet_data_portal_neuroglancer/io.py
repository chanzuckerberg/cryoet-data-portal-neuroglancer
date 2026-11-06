"""Input of OME-Zarr images, GLB meshes and point annotations.

OME-Zarr images are read with zarr-python 3, which opens both Zarr v2 stores (OME-NGFF 0.4, metadata in
``.zattrs``) and Zarr v3 stores (OME-NGFF 0.5, metadata under ``attributes.ome`` in ``zarr.json``), including
sharded arrays.
"""

import json
import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import ndjson
import numpy as np
import trimesh
import zarr
from zarr.abc.store import Store
from zarr.storage import FsspecStore, LocalStore

if TYPE_CHECKING:
    from pathlib import Path

# Axes that some layouts add in front of the spatial axes (e.g. the 5-D [t, c, z, y, x] volumes of the cryoET
# profile). ``load_omezarr_data(squeeze=True)`` drops them when they have size 1.
NON_SPATIAL_AXES = ("t", "c")


def open_store(
    src: "str | os.PathLike[str] | Store",
    storage_options: dict[str, Any] | None = None,
) -> Store:
    """A read-only zarr store for a local path, a ``file://`` URL or a remote URL (``s3://``, ``https://``, ...)."""
    if isinstance(src, Store):
        return src
    path = os.fspath(src)
    if path.startswith("file://"):
        path = path[len("file://") :]
    elif "://" in path:
        return FsspecStore.from_url(path, storage_options=storage_options or {}, read_only=True)
    return LocalStore(path, read_only=True)


@dataclass(frozen=True)
class OmeZarrImage:
    """An OME-Zarr image group (OME-NGFF 0.4 or 0.5) and its first ``multiscales`` entry."""

    group: zarr.Group

    @cached_property
    def ome(self) -> dict[str, Any]:
        """The OME metadata: ``attributes["ome"]`` for NGFF 0.5, the top-level attributes for 0.4."""
        attrs = dict(self.group.attrs)
        return dict(attrs["ome"]) if "ome" in attrs else attrs  # type: ignore[arg-type]

    @property
    def zarr_format(self) -> int:
        return int(self.group.metadata.zarr_format)

    @cached_property
    def multiscales(self) -> dict[str, Any]:
        if "multiscales" not in self.ome:
            raise ValueError(f"{self.group.store} has no OME-NGFF multiscales metadata")
        return dict(self.ome["multiscales"][0])

    @property
    def ngff_version(self) -> str:
        return str(self.ome.get("version", self.multiscales.get("version", "0.4")))

    @property
    def axes(self) -> list[str]:
        return [a["name"] if isinstance(a, dict) else str(a) for a in self.multiscales["axes"]]

    @property
    def datasets(self) -> list[dict[str, Any]]:
        return list(self.multiscales["datasets"])

    def _dataset(self, level: int | str) -> dict[str, Any]:
        if isinstance(level, str):
            return next(d for d in self.datasets if d["path"] == level)
        return self.datasets[level]  # negative levels count from the coarsest

    def array(self, level: int | str = 0) -> zarr.Array:
        """The array of a level, by index into ``datasets`` (negative from the end) or by its path."""
        return self.group[self._dataset(level)["path"]]  # type: ignore[return-value]

    def _transform(self, level: int | str, kind: str) -> list[float] | None:
        for t in self._dataset(level).get("coordinateTransformations", []):
            if t.get("type") == kind:
                return [float(v) for v in t[kind]]
        return None

    def scale(self, level: int | str = 0) -> list[float]:
        return self._transform(level, "scale") or [1.0] * len(self.axes)

    def translation(self, level: int | str = 0) -> list[float]:
        return self._transform(level, "translation") or [0.0] * len(self.axes)

    @property
    def scale_factor(self) -> float | None:
        """netCDF/CF ``scale_factor`` of the level arrays (``value = add_offset + scale_factor * stored``)."""
        v = self.array(0).attrs.get("scale_factor")
        return None if v is None else float(v)  # type: ignore[arg-type]

    @property
    def add_offset(self) -> float | None:
        v = self.array(0).attrs.get("add_offset")
        return None if v is None else float(v)  # type: ignore[arg-type]

    @property
    def label_values(self) -> list[int] | None:
        """Label values of an image-label group, from wherever the writer put them: ``ome["image-label"]``
        (NGFF 0.5; top-level ``image-label`` in NGFF 0.4) or ``multiscales[0].metadata["image-label"]`` (older
        CryoET Data Portal masks)."""
        for meta in (self.ome.get("image-label"), self.multiscales.get("metadata", {}).get("image-label")):
            if meta and "colors" in meta:
                return [int(c["label-value"]) for c in meta["colors"]]
        return None


def open_omezarr(
    src: "str | os.PathLike[str] | Store | zarr.Group",
    *,
    storage_options: dict[str, Any] | None = None,
) -> OmeZarrImage:
    """Open an OME-Zarr image (Zarr v2 / NGFF 0.4 or Zarr v3 / NGFF 0.5) read-only."""
    group = src if isinstance(src, zarr.Group) else zarr.open_group(open_store(src, storage_options), mode="r")
    return OmeZarrImage(group)


def dask_chunks(array: zarr.Array, min_size: int = 256) -> tuple[int, ...]:
    """Dask chunks for a zarr array.

    Unsharded arrays keep their chunks. For sharded arrays, dask would otherwise use the inner chunk shape, i.e.
    one request per inner chunk; along each axis use ``min(shard, max(inner, min_size))``, rounded down to a
    multiple of the inner chunk, so every dask block lies within one shard.
    """
    shards = array.shards
    if shards is None:
        return tuple(array.chunks)
    out = []
    for inner, shard in zip(array.chunks, shards, strict=True):
        size = min(shard, max(inner, min_size))
        out.append(max(inner, size // inner * inner))
    return tuple(out)


def decode_cf(data: Any, scale_factor: float | None, add_offset: float | None) -> Any:
    """Apply netCDF/CF scaling, ``value = add_offset + scale_factor * stored``, in float64; return float32."""
    if scale_factor is None and add_offset is None:
        return data
    sf = 1.0 if scale_factor is None else scale_factor
    ao = 0.0 if add_offset is None else add_offset
    return (data.astype(np.float64) * sf + ao).astype(np.float32)


def load_omezarr_data(
    input_filepath: "str | os.PathLike[str] | Store | zarr.Group",
    resolution_level: int | str = 0,
    persist: bool = True,
    *,
    storage_options: dict[str, Any] | None = None,
    chunks: Literal["auto", "native"] | tuple[int, ...] = "auto",
    squeeze: bool = True,
    decode: bool = False,
) -> da.Array:
    """Load one level of an OME-Zarr image as a dask array

    Parameters
    ----------
        input_filepath: str
            Path or URL of the OME-Zarr image (Zarr v2 / OME-NGFF 0.4 or Zarr v3 / OME-NGFF 0.5),
            or an open zarr store or group.
        resolution_level: int or str, optional
            Level to load, as an index into the multiscales datasets (negative counts from the
            lowest resolution) or as the dataset path. By default 0 - the highest resolution.
        persist: bool, optional
            Whether to persist the dask array.
            By default True.
        storage_options: dict, optional
            Options for fsspec, e.g. ``{"anon": True}`` for public S3 buckets.
        chunks: "auto", "native" or tuple, optional
            Dask chunks. "auto" keeps the chunks of unsharded arrays and uses shard-aligned blocks of
            at least 256 voxels per axis for sharded arrays; "native" uses the zarr chunks as they are.
        squeeze: bool, optional
            Drop ``t`` and ``c`` axes of size 1, so 5-D [t, c, z, y, x] volumes load as 3-D.
            By default True.
        decode: bool, optional
            Apply the CF ``scale_factor``/``add_offset`` attributes of the arrays (returns float32).
            By default False: the stored values are returned.

    Returns
    -------
        dask.array.Array
            The loaded OME-Zarr data as a dask array.
    """
    image = open_omezarr(input_filepath, storage_options=storage_options)
    array = image.array(resolution_level)
    if chunks == "auto":
        chunks = dask_chunks(array)
    elif chunks == "native":
        chunks = tuple(array.chunks)
    data = da.from_zarr(array, chunks=chunks)
    if squeeze:
        index = tuple(
            0 if (name in NON_SPATIAL_AXES and size == 1) else slice(None)
            for name, size in zip(image.axes, data.shape, strict=True)
        )
        data = data[index]
    if decode:
        data = decode_cf(data, image.scale_factor, image.add_offset)
    return data.persist() if persist else data


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
