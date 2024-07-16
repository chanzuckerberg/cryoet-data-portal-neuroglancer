import json
from functools import partial
from pathlib import Path
from typing import Iterator, Optional, Tuple

import DracoPy
import igneous.tasks.mesh.multires
import numpy as np
import shardcomputer
import trimesh
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume, Mesh
from cloudvolume.datasource.precomputed.mesh.multilod import (
    MultiLevelPrecomputedMeshManifest,
    to_stored_model_space,
)
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from igneous.task_creation.common import compute_shard_params_for_hashed
from igneous.task_creation.mesh import configure_multires_info
from igneous.tasks.mesh.multires import create_mesh_shard, create_octree_level_from_mesh, generate_lods, process_mesh
from taskqueue import LocalTaskQueue, queueable
from tqdm import tqdm


def process_decimated_mesh(
    cv: CloudVolume,
    label: int,
    meshes: list[Mesh],
    num_lod: int,
    min_chunk_size: Tuple[int, int, int] = (512, 512, 512),
    draco_compression_level: int = 7,
) -> Tuple[MultiLevelPrecomputedMeshManifest, Mesh]:
    grid_origin = np.floor(np.min(meshes[0].vertices, axis=0))
    mesh_shape = (np.max(meshes[0].vertices, axis=0) - grid_origin).astype(int)

    if np.any(mesh_shape == 0):
        return (None, None)

    max_lod = len(meshes)  # This is the number of LODs
    lods = meshes
    chunk_shape = np.ceil(mesh_shape / 2 ** (max_lod - 1))
    print(
        f"Processing data into {[int(c) for c in chunk_shape]} sized chunks for a {mesh_shape} size mesh grid with {max_lod} LODs",
    )

    if np.any(chunk_shape == 0):
        return (None, None)

    lods = [
        create_octree_level_from_mesh(lods[lod], chunk_shape, lod, len(lods))
        for lod in tqdm(range(len(lods)), desc="Processing LODs into octree")
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

    print(f"Encoding octree containing {len(lods)} LODs with Draco")
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

    old_process_mesh = process_mesh
    igneous.tasks.mesh.multires.process_mesh = process_decimated_mesh
    fname, shard = create_mesh_shard(
        cv,
        labels,
        num_lod,
        draco_compression_level,
        progress,
        shard_no,
        min_chunk_size,
    )
    igneous.tasks.mesh.multires.process_mesh = old_process_mesh

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


def decimate_mesh(
    mesh: trimesh.Trimesh,
    num_lods: int,
    aggressiveness: float = 7.0,
    as_trimesh: bool = False,
) -> list[Mesh] | list[trimesh.Trimesh]:
    """
    Decimate a mesh by a factor

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to decimate
    num_lods : int
        The number of levels of detail to generate
    """
    unused = 0
    lods = generate_lods(unused, mesh, num_lods, aggressiveness=aggressiveness)
    if as_trimesh:
        return [
            trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            for mesh in lods
            if not isinstance(mesh, trimesh.Trimesh)
        ]
    return lods


def _determine_mesh_shape(mesh: trimesh.Trimesh):
    # As resolution is fixed at 1.0, we don't need to care about it
    # See igneous/tasks/mesh/multires.py for the full function
    grid_origin = np.floor(np.min(mesh.vertices, axis=0))
    mesh_shape = (np.max(mesh.vertices, axis=0) - grid_origin).astype(int)
    return mesh_shape


def determine_chunk_size_for_lod(
    mesh_shape: tuple[int, int, int],
    min_lod: int,
    max_lod: int,
    min_chunk_dim: int,
):
    """
    Determine the chunk size for a given mesh shape and LOD levels

    Parameters
    ----------
    mesh_shape : tuple[int, int, int]
        The shape of the mesh
    min_lod : int
        The minimum required levels of detail
    max_lod : int
        The maximum desired levels of detail
    min_chunk_dim : int
        If the chunk size is smaller than this, it will be increased
        This means that the chunk size will be at least min_chunk_dim x min_chunk_dim x min_chunk_dim
        This can result in not actually reaching the desired min LOD levels

    Returns
    -------
    tuple[int, int, int]
        The chunk size
    """

    def _determine_chunk_shape_for_lod(lod_level):
        return 2 ** np.floor(np.log2(mesh_shape / 2**lod_level))

    # Find a power of 2 chunk size such that the minimum LOD is at least min_lod
    chunk_shape = _determine_chunk_shape_for_lod(min_lod)

    # If the chunk size is smaller than the minimum chunk dimension
    # then we can't respect that min_lod and need to increase in size
    if np.any(chunk_shape < min_chunk_dim):
        chunk_shape = 2 ** np.floor(np.log2(mesh_shape / min_chunk_dim))
    else:
        # If all of the chunk dimensions have not gone below the minimum chunk dimension
        # then we might be able to use a smaller chunk, up to the max lod
        for lod_level in range(min_lod, max_lod + 1):
            if np.all(chunk_shape / 2 > min_chunk_dim):
                chunk_shape = _determine_chunk_shape_for_lod(lod_level)
    return tuple([int(x) for x in chunk_shape.astype(int)])


def generate_sharded_mesh_from_lods(
    lods: list[trimesh.Scene],
    outfolder: str | Path,
    max_faces: int = int(2 * 1e6),
    label: int = 1,
    size: tuple[float, float, float] | None = None,
):
    lods = [lod.dump(concatenate=True) for lod in lods]

    # Find the first LOD that has less than max_faces
    found = False
    for first_lod, lod in enumerate(lods):
        if len(lod.faces) < max_faces:
            found = True
            break
    if not found:
        raise ValueError("No LODs have less than the maximum number of faces")
    num_lod = len(lods) - first_lod
    print(
        f"Using LOD {first_lod} as the first LOD, with {len(lods[first_lod].faces)} faces, which is less than {max_faces} maximum faces. There are {num_lod} LODs remaining in total.",
    )

    mesh = lods[first_lod]
    _, bb2 = lods[first_lod].bounds

    def _compute_size():
        max_bound = np.ceil(bb2)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = size if size is not None else _compute_size()

    mesh_shape = _determine_mesh_shape(mesh)
    smallest_chunk_size = determine_chunk_size_for_lod(
        mesh_shape,
        num_lod,
        num_lod,
        1,
    )

    # The resolution is not handled here, but in the neuroglancer state
    generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        resolution=1.0,
        mesh_chunk_size=smallest_chunk_size,
    )

    tq = LocalTaskQueue(progress=False)
    tasks = create_sharded_multires_mesh_tasks_from_glb(
        f"precomputed://file://{outfolder}",
        labels={label: lods[first_lod:]},
        mesh_dir="mesh",
        num_lod=num_lod,
        min_chunk_size=smallest_chunk_size,
    )
    tq.insert(tasks)
    tq.execute(progress=False)


def generate_standalone_sharded_multiresolution_mesh(
    glb: trimesh.Scene | str | Path,
    outfolder: str | Path,
    label: int = 1,
    size: tuple[float, float, float] | None = None,
    min_lod: int = 2,
    max_lod: int = 5,
    min_chunk_dim: int = 16,
):
    """
    Generate a standalone sharded multiresolution mesh from a glb file or scene

    Parameters
    ----------
    glb : trimesh.Scene | str | Path
        The glb file or scene to generate the mesh from
    outfolder : str | Path
        The output folder
    label : int, optional
        The label to use, by default 1
    size : tuple[float, float, float] | None, optional
        The size of the mesh bounding box, by default None
    min_lod : int, optional
        The minimum required level of detail, by default 2
        This would give three levels of detail, 0, 1, and 2
    max_lod : int, optional
        The maximum desired level of detail, by default 5
    min_chunk_dim : int, optional
        If the chunk size is smaller than this, it will be increased, by default 8
        This means that the chunk size will be at least 8x8x8
        This can result in not actually reaching the desired min LOD levels
    """
    scene: trimesh.Scene = trimesh.load(glb, force="scene") if isinstance(glb, (str, Path)) else glb
    mesh = scene.dump(concatenate=True)
    _, bb2 = mesh.bounds

    def _compute_size():
        max_bound = np.ceil(bb2)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = size if size is not None else _compute_size()

    mesh_shape = _determine_mesh_shape(mesh)
    smallest_chunk_size = determine_chunk_size_for_lod(
        mesh_shape,
        min_lod,
        max_lod,
        min_chunk_dim,
    )
    min_chunk_size = np.array(smallest_chunk_size, dtype=int)
    computed_max_lod = int(max(np.min(np.log2(mesh_shape / min_chunk_size)), 0))

    # The resolution is not handled here, but in the neuroglancer state
    generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        resolution=1.0,
        mesh_chunk_size=smallest_chunk_size,
    )

    tq = LocalTaskQueue()
    tasks = create_sharded_multires_mesh_tasks_from_glb(
        f"precomputed://file://{outfolder}",
        labels={label: mesh},
        mesh_dir="mesh",
        num_lod=computed_max_lod,
        min_chunk_size=smallest_chunk_size,
    )
    tq.insert(tasks)
    tq.execute()
