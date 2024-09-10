import functools
import os
import re
from collections import defaultdict
from typing import Dict, List

import DracoPy
import fastremap
import numpy as np
import simdjson
import trimesh
import zmesh
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume, Mesh
from cloudvolume.datasource.precomputed.mesh.multilod import (
    MultiLevelPrecomputedMeshManifest,
    to_stored_model_space,
)
from cloudvolume.lib import Bbox, Vec, toiter
from igneous.tasks.mesh.multires import cmp_zorder, generate_lods
from mapbuffer import MapBuffer

from cryoet_data_portal_neuroglancer.utils import determine_mesh_shape_from_lods


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


def process_mesh_into_octree_submeshes(
    mesh: trimesh.Trimesh,
    grid_origin: "Vec",
    grid_shape: "Vec",
    grid_scale: "Vec",
):
    nx, ny, nz = np.eye(3)
    ox, oy, oz = grid_origin * np.eye(3)
    submeshes = []
    nodes = []

    for x in range(0, grid_shape.x):
        # list(...) required b/c it doesn't like Vec classes
        mesh_x = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=nx,
            plane_origin=list(nx * x * grid_scale.x + ox),
            cap=False,
        )
        mesh_x = trimesh.intersections.slice_mesh_plane(
            mesh_x,
            plane_normal=-nx,
            plane_origin=list(nx * (x + 1) * grid_scale.x + ox),
            cap=False,
        )
        for y in range(0, grid_shape.y):
            mesh_y = trimesh.intersections.slice_mesh_plane(
                mesh_x,
                plane_normal=ny,
                plane_origin=list(ny * y * grid_scale.y + oy),
                cap=False,
            )
            mesh_y = trimesh.intersections.slice_mesh_plane(
                mesh_y,
                plane_normal=-ny,
                plane_origin=list(ny * (y + 1) * grid_scale.y + oy),
                cap=False,
            )
            for z in range(0, grid_shape.z):
                mesh_z = trimesh.intersections.slice_mesh_plane(
                    mesh_y,
                    plane_normal=nz,
                    plane_origin=list(nz * z * grid_scale.z + oz),
                    cap=False,
                )
                mesh_z = trimesh.intersections.slice_mesh_plane(
                    mesh_z,
                    plane_normal=-nz,
                    plane_origin=list(nz * (z + 1) * grid_scale.z + oz),
                    cap=False,
                )

                if len(mesh_z.vertices) == 0:
                    continue

                # test for totally degenerate meshes by checking if
                # all of two axes match, meaning the mesh must be a
                # point or a line.
                if np.sum([np.all(mesh_z.vertices[:, i] == mesh_z.vertices[0, i]) for i in range(3)]) >= 2:
                    continue

                submeshes.append(mesh_z)
                nodes.append((x, y, z))

    # Sort in Z-curve order
    submeshes, nodes = zip(
        *sorted(
            zip(submeshes, nodes, strict=True),
            key=functools.cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1])),
        ),
        strict=True,
    )
    # convert back from trimesh to CV Mesh class
    submeshes = [Mesh(m.vertices, m.faces) for m in submeshes]

    return (submeshes, nodes)


def retriangulate_mesh(mesh: trimesh.Trimesh, grid_origin: "Vec", grid_shape: "Vec", grid_scale: "Vec"):
    """
    Retriangulate the input mesh to avoid any cases where the boundaries of a triangle are split across the boundaries of the submeshes
    """
    nx, ny, nz = np.eye(3)
    ox, oy, oz = grid_origin * np.eye(3)
    new_mesh = trimesh.Trimesh()

    for x in range(0, grid_shape.x):
        # list(...) required b/c it doesn't like Vec classes
        mesh_x = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=nx,
            plane_origin=list(nx * x * grid_scale.x + ox),
            cap=False,
        )
        mesh_x = trimesh.intersections.slice_mesh_plane(
            mesh_x,
            plane_normal=-nx,
            plane_origin=list(nx * (x + 1) * grid_scale.x + ox),
            cap=False,
        )
        for y in range(0, grid_shape.y):
            mesh_y = trimesh.intersections.slice_mesh_plane(
                mesh_x,
                plane_normal=ny,
                plane_origin=list(ny * y * grid_scale.y + oy),
                cap=False,
            )
            mesh_y = trimesh.intersections.slice_mesh_plane(
                mesh_y,
                plane_normal=-ny,
                plane_origin=list(ny * (y + 1) * grid_scale.y + oy),
                cap=False,
            )
            for z in range(0, grid_shape.z):
                mesh_z = trimesh.intersections.slice_mesh_plane(
                    mesh_y,
                    plane_normal=nz,
                    plane_origin=list(nz * z * grid_scale.z + oz),
                    cap=False,
                )
                mesh_z = trimesh.intersections.slice_mesh_plane(
                    mesh_z,
                    plane_normal=-nz,
                    plane_origin=list(nz * (z + 1) * grid_scale.z + oz),
                    cap=False,
                )

                if len(mesh_z.vertices) == 0:
                    continue

                # test for totally degenerate meshes by checking if
                # all of two axes match, meaning the mesh must be a
                # point or a line.
                if np.sum([np.all(mesh_z.vertices[:, i] == mesh_z.vertices[0, i]) for i in range(3)]) >= 2:
                    continue

                new_mesh = trimesh.util.concatenate(new_mesh, mesh_z)
    return new_mesh


def patched_create_octree_level_from_mesh(mesh, chunk_shape, lod, num_lods, grid_origin, grid_length):
    """
    Create submeshes by slicing the orignal mesh to produce smaller chunks
    by slicing them from x,y,z dimensions.

    This creates (2^lod)^3 submeshes.
    """

    grid_scale = Vec(*(np.array(chunk_shape) * (2**lod)))
    grid_shape = Vec(*(np.ceil(grid_length / grid_scale)), dtype=int)
    #print(f"grid_origin: {grid_origin}, grid_scale: {grid_scale}, grid_shape: {grid_shape}")
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # If not LOD 0 need to retriangulate the input mush to avoid any cases where
    # the boundaries of a triangle are split across the boundaries of the submeshes
    # at the higher level of the octree
    if lod > 0:
        upper_grid_scale = Vec(*(np.array(chunk_shape) * (2 ** (lod - 1))))
        upper_grid_shape = Vec(*np.ceil(grid_length / upper_grid_scale), dtype=int)
        mesh = retriangulate_mesh(mesh, grid_origin, upper_grid_shape, upper_grid_scale)

    if lod != num_lods - 1:
        return process_mesh_into_octree_submeshes(mesh, grid_origin, grid_shape, grid_scale)
    return ([Mesh(mesh.vertices, mesh.faces)], ((0, 0, 0),))


def patched_process_mesh(
    cv: CloudVolume,
    label: int,
    mesh: Mesh,
    num_lod: int,
    min_chunk_size=(512, 512, 512),
    draco_compression_level: int = 7,
):
    mesh.vertices /= cv.meta.resolution(cv.mesh.meta.mip)
    grid_origin = np.floor(np.min(mesh.vertices, axis=0))
    mesh_shape = (np.max(mesh.vertices, axis=0) - grid_origin).astype(int)

    if np.any(mesh_shape == 0):
        return (None, None)

    min_chunk_size = np.array(min_chunk_size, dtype=int)
    max_lod = int(max(np.min(np.log2(mesh_shape / min_chunk_size)), 0))
    max_lod = min(max_lod, num_lod)

    lods = generate_lods(label, mesh, max_lod)
    grid_origin, mesh_shape = determine_mesh_shape_from_lods(lods)
    chunk_shape = np.ceil(mesh_shape / (2 ** (len(lods) - 1)))

    if np.any(chunk_shape == 0):
        return (None, None)

    lods = [
        patched_create_octree_level_from_mesh(lods[lod], chunk_shape, lod, len(lods), grid_origin, mesh_shape)
        for lod in range(len(lods))
    ]
    fragment_positions = [nodes for submeshes, nodes in lods]
    lods = [submeshes for submeshes, nodes in lods]

    manifest = MultiLevelPrecomputedMeshManifest(
        segment_id=label,
        chunk_shape=chunk_shape,
        grid_origin=grid_origin,
        num_lods=len(lods),
        lod_scales=[2**i for i in range(len(lods))],
        vertex_offsets=[[0, 0, 0]] * len(lods),
        num_fragments_per_lod=[len(lods[lod]) for lod in range(len(lods))],
        fragment_positions=fragment_positions,
        fragment_offsets=[],  # needs to be set when we have the final value
    )

    vqb = int(cv.mesh.meta.info["vertex_quantization_bits"])

    mesh_binaries = []
    for lod, submeshes in enumerate(lods):
        for frag_no, submesh in enumerate(submeshes):
            submesh.vertices = to_stored_model_space(
                submesh.vertices,
                manifest,
                lod=lod,
                vertex_quantization_bits=vqb,
                frag=frag_no,
            )

            minpt = np.min(submesh.vertices, axis=0)
            quantization_range = np.max(submesh.vertices, axis=0) - minpt
            quantization_range = np.max(quantization_range)

            # mesh.vertices must be integer type or mesh will display
            # distorted in neuroglancer.
            try:
                submesh = DracoPy.encode(
                    submesh.vertices,
                    submesh.faces,
                    quantization_bits=vqb,
                    compression_level=draco_compression_level,
                    quantization_range=quantization_range,
                    quantization_origin=minpt,
                    create_metadata=True,
                )
            except DracoPy.EncodingFailedException:
                submesh = b""

            manifest.fragment_offsets.append(len(submesh))
            mesh_binaries.append(submesh)

    return (manifest, b"".join(mesh_binaries))


def patch():
    import cloudvolume.datasource.precomputed.spatial_index
    import igneous.tasks.mesh.mesh
    import igneous.tasks.mesh.multires

    igneous.tasks.mesh.multires.locations_for_labels = patched_locations_for_labels
    igneous.tasks.mesh.multires.process_mesh = patched_process_mesh
    igneous.tasks.mesh.mesh.MeshTask.execute = PatchedMeshTask_execute
    igneous.tasks.mesh.mesh.MeshTask._upload_batch = PatchedMeshTask_upload_batch
    cloudvolume.datasource.precomputed.spatial_index.SpatialIndex.file_locations_per_label_json = (
        patched_file_locations_per_label_json
    )
