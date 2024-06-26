import os
import re
from collections import defaultdict
from typing import Dict, List

import fastremap
import numpy as np
import simdjson
import zmesh
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec, toiter
from mapbuffer import MapBuffer


def PatchedMeshTask_execute(self):  # noqa
    self._volume = CloudVolume(
        self.layer_path,
        self.options["mip"],
        bounded=False,
        parallel=self.options["parallel_download"],
        fill_missing=self.options["fill_missing"],
    )
    resolution = self._volume.resolution
    self._bounds = Bbox(self.offset, self.shape + self.offset, dtype=resolution.dtype)
    self._bounds = Bbox.clamp(self._bounds, self._volume.bounds)

    self.progress = bool(self.options["progress"])

    self._mesher = zmesh.Mesher(self._volume.resolution)

    # Marching cubes loves its 1vx overlaps.
    # This avoids lines appearing between
    # adjacent chunks.
    data_bounds = self._bounds.clone()
    data_bounds.minpt -= self.options["low_padding"]
    data_bounds.maxpt += self.options["high_padding"]

    self._mesh_dir = self.get_mesh_dir()

    if self.options["encoding"] == "draco":
        self.draco_encoding_settings = draco_encoding_settings(  # noqa
            shape=(self.shape + self.options["low_padding"] + self.options["high_padding"]),
            offset=self.offset,
            resolution=self._volume.resolution,
            compression_level=self.options["draco_compression_level"],
            create_metadata=self.options["draco_create_metadata"],
            uses_new_draco_bin_size=False,
        )

    # chunk_position includes the overlap specified by low_padding/high_padding
    # agglomerate, timestamp, stop_layer only applies to graphene volumes,
    # no-op for precomputed
    data = self._volume.download(
        data_bounds,
        agglomerate=self.options["agglomerate"],
        timestamp=self.options["timestamp"],
        stop_layer=self.options["stop_layer"],
    )

    if not np.any(data):
        if self.options["spatial_index"]:
            self._upload_spatial_index(self._bounds, {})
        return

    left_offset = Vec(0, 0, 0)
    if self.options["closed_dataset_edges"]:
        data, left_offset = self._handle_dataset_boundary(data, data_bounds)

    data = self._remove_dust(data, self.options["dust_threshold"], self.options["dust_global"])
    data = self._remap(data)

    if self.options["object_ids"]:
        data = fastremap.mask_except(data, self.options["object_ids"], in_place=True)

    data, renumbermap = fastremap.renumber(data, in_place=True)
    renumbermap = {v: k for k, v in renumbermap.items()}

    self._mesher.mesh(data[..., 0].T)
    del data

    self.compute_meshes(renumbermap, left_offset)


def PatchedMeshTask_upload_batch(self, meshes, bbox):  # noqa

    frag_path = self.options["frag_path"] or self.layer_path
    cf = CloudFiles(frag_path, progress=self.options["progress"])

    mbuf = MapBuffer(meshes, compress="br")

    cf.put(
        f"{self._mesh_dir}/{bbox.to_filename(1)}.frags",
        content=mbuf.tobytes(),
        compress=None,
        content_type="application/x.mapbuffer",
        cache_control=False,
    )


def patched_locations_for_labels(cv: CloudVolume, labels: List[int]) -> Dict[int, List[str]]:

    SPATIAL_EXT = re.compile(r"\.spatial$")  # noqa
    index_filenames = cv.mesh.spatial_index.file_locations_per_label(labels, allow_missing=True)
    resolution = cv.meta.resolution(cv.mesh.meta.mip)
    for label, locations in index_filenames.items():
        for i, location in enumerate(locations):
            bbx = Bbox.from_filename(re.sub(SPATIAL_EXT, "", location), dtype=resolution.dtype)
            bbx /= resolution

            index_filenames[label][i] = bbx.to_filename(1) + ".frags"
    return index_filenames


def patched_file_locations_per_label_json(self, labels, allow_missing=False):
    locations = defaultdict(list)
    parser = simdjson.Parser()

    if labels is not None:
        labels = set(toiter(labels))

    for index_files in self.fetch_all_index_files(allow_missing=allow_missing):
        for filename, content in index_files.items():
            if not content:
                continue
            index_labels = set(parser.parse(content).keys())
            filename = os.path.basename(filename)

            if labels is None:
                for label in index_labels:
                    locations[int(label)].append(filename)
            elif len(labels) > len(index_labels):
                for label in index_labels:
                    if int(label) in labels:
                        locations[int(label)].append(filename)
            else:
                for label in labels:
                    if str(label) in index_labels:
                        locations[int(label)].append(filename)

    return locations


def patch():
    import cloudvolume.datasource.precomputed.spatial_index
    import igneous.tasks.mesh.mesh
    import igneous.tasks.mesh.multires

    igneous.tasks.mesh.multires.locations_for_labels = patched_locations_for_labels
    igneous.tasks.mesh.mesh.MeshTask.execute = PatchedMeshTask_execute
    igneous.tasks.mesh.mesh.MeshTask._upload_batch = PatchedMeshTask_upload_batch
    cloudvolume.datasource.precomputed.spatial_index.SpatialIndex.file_locations_per_label_json = (
        patched_file_locations_per_label_json
    )
