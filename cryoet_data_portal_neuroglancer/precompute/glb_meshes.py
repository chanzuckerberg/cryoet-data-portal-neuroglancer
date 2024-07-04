import json
from functools import partial
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import shardcomputer
import trimesh
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume, Mesh
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from igneous.task_creation.common import compute_shard_params_for_hashed
from igneous.task_creation.mesh import configure_multires_info
from igneous.tasks.mesh.multires import create_mesh_shard
from taskqueue import LocalTaskQueue, queueable


@queueable
def MultiResShardedMeshFromGlbTask(  # noqa
    cloudpath: str,
    shard_no: str,
    labels: dict[int, Mesh],
    draco_compression_level: int = 1,
    mesh_dir: Optional[str] = None,
    cache: Optional[bool] = False,
    num_lod: int = 1,
    spatial_index_db: Optional[str] = None,
    min_chunk_size: tuple[int, int, int] = (128, 128, 128),
    progress: bool = False,
):
    cv = CloudVolume(cloudpath, spatial_index_db=spatial_index_db, cache=cache)
    cv.mip = cv.mesh.meta.mip
    if mesh_dir is None and "mesh" in cv.info:
        mesh_dir = cv.info["mesh"]

    fname, shard = create_mesh_shard(cv, labels, num_lod, draco_compression_level, progress, shard_no, min_chunk_size)

    if shard is None:
        return

    cf = CloudFiles(cv.mesh.meta.layerpath)
    cf.put(
        fname,
        shard,
        compress=False,
        content_type="application/octet-stream",
        cache_control="no-cache",
    )


def create_sharded_multires_mesh_tasks_from_glb(
    cloudpath: str,
    labels: dict[int, Mesh],
    shard_index_bytes=2**13,
    minishard_index_bytes=2**15,
    min_shards: int = 1,
    num_lod: int = 0,
    draco_compression_level: int = 7,
    vertex_quantization_bits: int = 16,
    minishard_index_encoding="gzip",
    mesh_dir: Optional[str] = None,
    spatial_index_db: Optional[str] = None,
    cache: Optional[bool] = False,
    min_chunk_size: tuple[int, int, int] = (256, 256, 256),
    max_labels_per_shard: Optional[int] = None,
) -> Iterator[MultiResShardedMeshFromGlbTask]:

    mesh_info = configure_multires_info(cloudpath, vertex_quantization_bits, mesh_dir)

    # rebuild b/c sharding changes the mesh source class
    cv = CloudVolume(cloudpath, progress=True, spatial_index_db=spatial_index_db)
    cv.mip = cv.mesh.meta.mip

    all_labels = labels

    if max_labels_per_shard is not None:
        assert max_labels_per_shard >= 1
        min_shards = max(int(np.ceil(len(all_labels) / max_labels_per_shard)), min_shards)

    (shard_bits, minishard_bits, preshift_bits) = compute_shard_params_for_hashed(
        num_labels=len(all_labels),
        shard_index_bytes=int(shard_index_bytes),
        minishard_index_bytes=int(minishard_index_bytes),
        min_shards=min_shards,
    )

    spec = ShardingSpecification(
        type="neuroglancer_uint64_sharded_v1",
        preshift_bits=preshift_bits,
        hash="murmurhash3_x86_128",
        minishard_bits=minishard_bits,
        shard_bits=shard_bits,
        minishard_index_encoding=minishard_index_encoding,
        data_encoding="raw",  # draco encoded meshes
    )

    cv.mesh.meta.info = mesh_info  # ensure no race conditions
    cv.mesh.meta.info["sharding"] = spec.to_dict()
    cv.mesh.meta.commit_info()

    cv = CloudVolume(cloudpath)

    all_labels = np.fromiter(all_labels, dtype=np.uint64, count=len(all_labels))
    shard_labels = shardcomputer.assign_labels_to_shards(all_labels, preshift_bits, shard_bits, minishard_bits)
    del all_labels

    # ?
    # cf = CloudFiles(cv.mesh.meta.layerpath, progress=True)
    # files = ((f"{shardno}.labels", labels) for shardno, labels in shard_labels.items())
    # cf.put_jsons(files, compress="gzip", cache_control="no-cache", total=len(shard_labels))

    return [
        partial(
            MultiResShardedMeshFromGlbTask,
            cloudpath,
            shard_no,
            labels=labels,
            num_lod=num_lod,
            mesh_dir=mesh_dir,
            cache=cache,
            spatial_index_db=spatial_index_db,
            draco_compression_level=draco_compression_level,
            min_chunk_size=min_chunk_size,
        )
        for shard_no in shard_labels
    ]


def generate_standalone_mesh_info(
    outfolder: str | Path,
    size: tuple[float, float, float] | float,
    mesh_dir: str = "mesh",
    resolution: tuple[float, float, float] | float = (1.0, 1.0, 1.0),
    mesh_chunk_size: tuple[float, float, float] | float = (448, 448, 448),
):
    outfolder = Path(outfolder)
    outfolder.mkdir(exist_ok=True, parents=True)
    resolution_conv = resolution if isinstance(resolution, tuple) else (resolution,) * 3
    mesh_chunk_size_conv = mesh_chunk_size if isinstance(mesh_chunk_size, tuple) else (mesh_chunk_size,) * 3

    # offset = bbox.transform[:, 3][:3].tolist()
    info = outfolder / "info"
    info.write_text(
        json.dumps(
            {
                "@type": "neuroglancer_multiscale_volume",
                "data_type": "uint32",
                "mesh": "mesh",
                "num_channels": 1,
                "scales": [
                    {
                        "chunk_sizes": [[256, 256, 256]],  # information required by neuroglancer but not used
                        "compressed_segmentation_block_size": [
                            64,
                            64,
                            64,
                        ],  # information required by neuroglancer but not used
                        "encoding": "compressed_segmentation",
                        "key": "data",
                        "resolution": resolution_conv,
                        "size": size,
                    },
                ],
                "type": "segmentation",
            },
            indent=2,
        ),
    )

    mesh_info = outfolder / mesh_dir
    mesh_info.mkdir(exist_ok=True, parents=True)

    mesh_info /= "info"
    mesh_info.write_text(
        json.dumps(
            {
                "@type": "neuroglancer_multilod_draco",
                "mip": 0,
                "chunk_size": mesh_chunk_size_conv,
                "spatial_index": {"resolution": [1, 1, 1], "chunk_size": mesh_chunk_size_conv},
                "vertex_quantization_bits": 16,
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                "lod_scale_multiplier": 1.0,
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "preshift_bits": 0,
                    "hash": "murmurhash3_x86_128",
                    "minishard_bits": 0,
                    "shard_bits": 0,
                    "minishard_index_encoding": "gzip",
                    "data_encoding": "raw",
                },
            },
            indent=2,
        ),
    )


def generate_standalone_sharded_multiresolution_mesh(
    glb_file: str | Path,
    outfolder: str | Path,
    label: int,
    size: tuple[float, float, float] | None = None,
    resolution: tuple[float, float, float] | float = (1.0, 1.0, 1.0),
):
    scene: trimesh.Scene = trimesh.load(glb_file)
    mesh = next(iter(scene.geometry.values()))
    bb1, bb2 = mesh.bounds
    size_x, size_y, size_z = size if size is not None else np.ceil(np.abs(bb2) - np.abs(bb1)) * resolution

    mesh.apply_translation((bb2 - bb1) / 2)

    generate_standalone_mesh_info(outfolder, size=(size_x, size_y, size_z), resolution=resolution)

    tq = LocalTaskQueue()
    tasks = create_sharded_multires_mesh_tasks_from_glb(
        f"precomputed://file://{outfolder}",
        labels={label: mesh},
        mesh_dir="mesh",
        num_lod=10,
    )
    tq.insert(tasks)
    tq.execute()
