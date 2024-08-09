import json
import logging
from functools import partial
from pathlib import Path
from typing import Iterator, cast

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

LOGGER = logging.getLogger(__name__)


def _process_decimated_mesh(
    cv: CloudVolume,
    label: int,
    meshes: list[Mesh],
    num_lod: int,
    min_chunk_size: tuple[int, int, int] = (512, 512, 512),
    draco_compression_level: int = 7,
) -> tuple[MultiLevelPrecomputedMeshManifest, Mesh] | tuple[None, None]:
    grid_origin, mesh_shape = _determine_mesh_shape_from_lods(meshes)

    if np.any(mesh_shape == 0):
        return (None, None)

    num_lods = num_lod
    lods = meshes
    chunk_shape = np.ceil(mesh_shape / 2 ** (num_lods - 1))
    LOGGER.info(
        "Processing data into %s sized chunks for a %s size mesh grid with %i LODs",
        [int(c) for c in chunk_shape],
        mesh_shape,
        num_lods,
    )

    if np.any(chunk_shape == 0):
        return (None, None)

    # Igneous expects a Mesh object, but we have a trimesh object
    # At the moment, Igneous only actually needs this for the final
    # mesh - but for possible future compatibility, we convert all
    # meshes to Mesh objects
    lods = [Mesh(lod.vertices, lod.faces) for lod in lods]

    lods = [
        create_octree_level_from_mesh(lods[lod], chunk_shape, lod, num_lods)
        for lod in tqdm(range(num_lods), desc="Processing LODs into octree")
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

    LOGGER.info("Encoding octree containing %i LODs with Draco", len(lods))
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
    mesh_dir: str | None = None,
    cache: bool | None = False,
    num_lod: int = 1,
    spatial_index_db: str | None = None,
    min_chunk_size: tuple[int, int, int] = (128, 128, 128),
    progress: bool = False,
    use_decimated_mesh: bool = False,
):
    cv = CloudVolume(cloudpath, spatial_index_db=spatial_index_db, cache=cache)
    cv.mip = cv.mesh.meta.mip
    if mesh_dir is None and "mesh" in cv.info:
        mesh_dir = cv.info["mesh"]

    if use_decimated_mesh:
        old_process_mesh = process_mesh
        igneous.tasks.mesh.multires.process_mesh = _process_decimated_mesh
    fname, shard = create_mesh_shard(
        cv,
        labels,
        num_lod,
        draco_compression_level,
        progress,
        shard_no,
        min_chunk_size,
    )
    if use_decimated_mesh:
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


def _create_sharded_multires_mesh_tasks_from_glb(
    cloudpath: str,
    labels: dict[int, Mesh | trimesh.Trimesh | list[trimesh.Trimesh]],
    shard_index_bytes=2**13,
    minishard_index_bytes=2**15,
    min_shards: int = 1,
    num_lod: int = 0,
    draco_compression_level: int = 7,
    vertex_quantization_bits: int = 16,
    minishard_index_encoding="gzip",
    mesh_dir: str = "",
    spatial_index_db: str | None = None,
    cache: bool | None = False,
    min_chunk_size: tuple[int, int, int] = (256, 256, 256),
    max_labels_per_shard: int | None = None,
    use_decimated_mesh: bool = False,
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
            use_decimated_mesh=use_decimated_mesh,
        )
        for shard_no in shard_labels
    ]


def _generate_standalone_mesh_info(
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
    LOGGER.debug("Generating mesh info with chunk size %s", mesh_chunk_size_conv)

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
    aggressiveness: float = 5.5,
    as_trimesh: bool = False,
) -> list[Mesh] | list[trimesh.Trimesh]:
    """
    Decimate a mesh by a factor

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to decimate
    num_lods : int
        The number of levels of detail after decimation
        The number of generated LODs will be num_lods - 1
    aggressiveness : float, optional
        The aggressiveness of the decimation algorithm, by default 5.5
    as_trimesh : bool, optional
        Whether to return the mesh as a trimesh or a mesh, by default False
    """
    unused = 0
    # The num LODs here is the number of LODs to generate, not the number of LODs to use
    # So that is why we subtract 1
    lods = generate_lods(unused, mesh, num_lods - 1, aggressiveness=aggressiveness)  # type: ignore
    total_faces_per_lod = [len(lod.faces) for lod in lods]
    last_lod = len(lods) - 1
    for i in range(last_lod):
        if total_faces_per_lod[i] == total_faces_per_lod[i + 1]:
            last_lod = i
            break
    LOGGER.debug("Decimated mesh has %i LODs, with %s faces", last_lod + 1, total_faces_per_lod[: last_lod + 1])
    if as_trimesh:
        result = []
        for lod in lods[: last_lod + 1]:
            if not isinstance(lod, trimesh.Trimesh):
                lod = trimesh.Trimesh(vertices=lod.vertices, faces=lod.faces)
            result.append(lod)
        return result
    return lods[: last_lod + 1]


def _determine_mesh_shape(mesh: trimesh.Trimesh):
    # As resolution is fixed at 1.0, we don't need to care about it
    # See igneous/tasks/mesh/multires.py for the full function
    grid_origin = np.floor(np.min(mesh.vertices, axis=0))
    mesh_shape = (np.max(mesh.vertices, axis=0) - grid_origin).astype(int)
    return mesh_shape


def _determine_mesh_shape_from_lods(lods: list[trimesh.Trimesh]):
    mesh_starts = [np.min(lod.vertices, axis=0) for lod in lods]
    mesh_ends = [np.max(lod.vertices, axis=0) for lod in lods]
    LOGGER.debug(
        "LOD mesh origin points %s and end points %s",
        mesh_starts,
        mesh_ends,
    )
    grid_origin = np.floor(np.min(mesh_starts, axis=0))
    grid_end = np.max(mesh_ends, axis=0)
    mesh_shape = (grid_end - grid_origin).astype(int)
    return grid_origin, mesh_shape


def determine_chunk_size_for_lod(
    mesh_shape: tuple[int, int, int],
    max_lod: int,
    min_chunk_dim: int = 16,
) -> tuple[int, int, int]:
    """
    Determine the chunk size for a given mesh shape and LOD levels

    The multi-resolution mesh is generated by decimating the mesh
    by a factor of 2 for each level of detail. The decimated meshes
    are placed into an octree structure to represent the LODs.
    This function determines the needed chunk size for the highest
    resolution mesh representation, which is the first LOD.

    Parameters
    ----------
    mesh_shape : tuple[int, int, int]
        The shape of the mesh
    max_lod: int
        The max level of detail to generate, starting at 0
    min_chunk_dim : int, optional
        Any chunk dimension won't be smaller than this.
        If the chunk size needs to be smaller than this to respect the max
        LODs then the total LODs will be reduced.
        By default, this value is 16. This is needed because otherwise,
        the decimated meshes are likely to have errors.

    Returns
    -------
    tuple[int, int, int], int
        The chunk size, and the number of LODs that can be generated
    """
    mesh_shape = np.array(mesh_shape)

    def _determine_chunk_shape_for_lod(lod_level):
        return 2 ** np.floor(np.log2(mesh_shape / 2**lod_level))

    # Find a power of 2 chunk size to reach the max LOD
    chunk_shape = _determine_chunk_shape_for_lod(max_lod)

    # If the chunk size is smaller than the minimum chunk dimension
    # then we can't respect the max LOD and need to reduce the LODs
    if np.any(chunk_shape < min_chunk_dim):
        max_lod = int(max(np.min(np.log2(mesh_shape / min_chunk_dim)), 0))
        chunk_shape = _determine_chunk_shape_for_lod(max_lod)
    final_lod = int(max(np.min(np.log2(mesh_shape / chunk_shape)), 0)) + 1
    LOGGER.info(
        "Will produce %i LODs for this mesh at min size %s",
        final_lod,
        chunk_shape,
    )
    x, y, z = chunk_shape.astype(int)
    return (int(x), int(y), int(z)), final_lod


def generate_mesh_from_lods(
    lods: list[trimesh.Scene],
    outfolder: str | Path,
    min_mesh_chunk_dim: int = 16,
    label: int = 1,
    bounding_box_size: tuple[float, float, float] | None = None,
):
    """
    Generate a sharded mesh from a list of LODs

    Parameters
    ----------
    lods : list[trimesh.Scene]
        The list of LODs to generate the mesh from
    outfolder : str | Path
        The output folder
    min_mesh_chunk_dim : int, optional
        The minimum chunk dimension, by default 16. This is needed because
        otherwise, the decimated meshes are likely to have errors.
        This can result in a subset of the LODs generated being used.
    label : int, optional
        The label to use, by default 1
    bounding_box_size : tuple[float, float, float] | None, optional
        The bounding box size, by default None
        When None, the bounding box size is determined from the mesh
        This calculation is often not accurate, so it is recommended to
        provide the bounding box size. Or turn off the bounding box in the
        neuroglancer state.
    """
    concatenated_lods: list[trimesh.Trimesh] = cast(list[trimesh.Trimesh], [lod.dump(concatenate=True) for lod in lods])
    num_lod = len(concatenated_lods)
    first_lod = 0
    _, bb2 = concatenated_lods[first_lod].bounds

    def _compute_size(bbx):
        max_bound = np.ceil(bbx)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = bounding_box_size if bounding_box_size is not None else _compute_size(bb2)

    _, mesh_shape = _determine_mesh_shape_from_lods(concatenated_lods)
    smallest_chunk_size, calculated_num_lod = determine_chunk_size_for_lod(
        mesh_shape,
        num_lod - 1,
        min_chunk_dim=min_mesh_chunk_dim,
    )
    # TODO get actual chunk shape in other places too
    actual_chunk_shape = np.ceil(mesh_shape / 2 ** (calculated_num_lod - 1))

    # The resolution is not handled here, but in the neuroglancer state
    _generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        resolution=1.0,
        mesh_chunk_size=tuple([int(x) for x in actual_chunk_shape]),
    )

    tq = LocalTaskQueue(progress=False)
    tasks = _create_sharded_multires_mesh_tasks_from_glb(
        f"precomputed://file://{outfolder}",
        labels={label: concatenated_lods[first_lod:]},
        mesh_dir="mesh",
        num_lod=calculated_num_lod,
        min_chunk_size=smallest_chunk_size,
        use_decimated_mesh=True,
    )
    tq.insert(tasks)
    tq.execute(progress=False)


def generate_multiresolution_mesh(
    glb: trimesh.Scene | str | Path,
    outfolder: str | Path,
    max_lod: int = 2,
    min_mesh_chunk_dim: int = 16,
    bounding_box_size: tuple[float, float, float] | None = None,
    label: int = 1,
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
    max_lod : int, optional
        The maximum desired level of detail, by default 2
        This would give 3 LODs, 0, 1, 2, high, medium, low
    min_chunk_dim : int, optional
        If the chunk size is smaller than this, it will be increased, by default 16
        This means that the chunk size will be at least 16x16x16
        This can result in not actually reaching the desired min LOD levels
    """
    scene: trimesh.Scene = (
        cast(trimesh.Scene, trimesh.load(glb, force="scene")) if isinstance(glb, (str, Path)) else glb
    )
    mesh: trimesh.Trimesh = scene.dump(concatenate=True)  # type: ignore
    _, bb2 = mesh.bounds  # type: ignore

    def _compute_size():
        max_bound = np.ceil(bb2)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = bounding_box_size if bounding_box_size is not None else _compute_size()

    mesh_shape = _determine_mesh_shape(mesh)
    smallest_chunk_size, computed_num_lod = determine_chunk_size_for_lod(
        mesh_shape,
        max_lod,
        min_mesh_chunk_dim,
    )

    # The resolution is not handled here, but in the neuroglancer state
    _generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        resolution=1.0,
        mesh_chunk_size=smallest_chunk_size,
    )

    tq = LocalTaskQueue()
    tasks = _create_sharded_multires_mesh_tasks_from_glb(
        f"precomputed://file://{outfolder}",
        labels={label: mesh},
        mesh_dir="mesh",
        num_lod=computed_num_lod,
        min_chunk_size=smallest_chunk_size,
        use_decimated_mesh=False,
    )
    tq.insert(tasks)
    tq.execute()


def generate_multilabel_multiresolution_mesh(
    label_dict: dict[int, trimesh.Scene | str | Path],
    outfolder: str | Path,
    max_lod: int = 2,
    min_mesh_chunk_dim: int = 16,
    bounding_box_size: tuple[float, float, float] | None = None,
):
    """
    Generate a standalone sharded multiresolution mesh from a glb file or scene

    Parameters
    ----------
    label_dict: dict[int, trimesh.Scene | str | Path]
        The dictionary of label to glb file or scene to generate the mesh from
    outfolder : str | Path
        The output folder
    size : tuple[float, float, float] | None, optional
        The size of the mesh bounding box, by default None
    max_lod : int, optional
        The maximum desired level of detail, by default 2
        This would give 3 LODs, 0, 1, 2, high, medium, low
    min_chunk_dim : int, optional
        If the chunk size is smaller than this, it will be increased, by default 16
        This means that the chunk size will be at least 16x16x16
        This can result in not actually reaching the desired min LOD levels
    """
    labels = {}
    for k, v in label_dict.items():
        scene: trimesh.Scene = cast(trimesh.Scene, trimesh.load(v, force="scene")) if isinstance(v, (str, Path)) else v
        mesh: trimesh.Trimesh = scene.dump(concatenate=True)  # type: ignore
        labels[k] = mesh

    mesh = list(labels.values())[0]
    _, bb2 = mesh.bounds  # type: ignore

    def _compute_size():
        max_bound = np.ceil(bb2)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = bounding_box_size if bounding_box_size is not None else _compute_size()

    mesh_shape = _determine_mesh_shape(mesh)
    smallest_chunk_size, computed_num_lod = determine_chunk_size_for_lod(
        mesh_shape,
        max_lod,
        min_mesh_chunk_dim,
    )

    # The resolution is not handled here, but in the neuroglancer state
    _generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        resolution=1.0,
        mesh_chunk_size=smallest_chunk_size,
    )

    tq = LocalTaskQueue()
    tasks = _create_sharded_multires_mesh_tasks_from_glb(
        f"precomputed://file://{outfolder}",
        labels=labels,
        mesh_dir="mesh",
        num_lod=computed_num_lod,
        min_chunk_size=smallest_chunk_size,
        use_decimated_mesh=False,
    )
    tq.insert(tasks)
    tq.execute()
