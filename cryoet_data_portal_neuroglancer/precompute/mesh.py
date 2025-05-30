import json
import logging
from functools import partial
from pathlib import Path
from typing import Iterator, cast

import dask.array as da
import DracoPy
import igneous.task_creation as tc
import igneous.tasks.mesh.multires
import neuroglancer
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

from cryoet_data_portal_neuroglancer.precompute.segmentation_properties import write_segment_properties
from cryoet_data_portal_neuroglancer.utils import determine_mesh_shape_from_lods

LOGGER = logging.getLogger(__name__)


def _process_decimated_mesh(
    cv: CloudVolume,
    label: int,
    meshes: list[Mesh],
    num_lod: int,
    min_chunk_size: tuple[int, int, int] = (512, 512, 512),
    draco_compression_level: int = 7,
) -> tuple[MultiLevelPrecomputedMeshManifest, Mesh] | tuple[None, None]:
    grid_origin, mesh_shape = determine_mesh_shape_from_lods(meshes)

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
        create_octree_level_from_mesh(lods[lod], chunk_shape, lod, num_lods, grid_origin, mesh_shape)
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
        original_process_mesh = process_mesh
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
        igneous.tasks.mesh.multires.process_mesh = original_process_mesh

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
    mesh_chunk_size: tuple[float, float, float] | float = (448, 448, 448),
    has_segment_properties: bool = False,
):
    outfolder = Path(outfolder)
    outfolder.mkdir(exist_ok=True, parents=True)
    resolution = (1.0, 1.0, 1.0)
    mesh_chunk_size_conv = mesh_chunk_size if isinstance(mesh_chunk_size, tuple) else (mesh_chunk_size,) * 3
    LOGGER.debug("Generating mesh info with chunk size %s", mesh_chunk_size_conv)

    # offset = bbox.transform[:, 3][:3].tolist()
    info = outfolder / "info"
    output = {
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
                "resolution": resolution,
                "size": size,
            },
        ],
        "type": "segmentation",
    }

    if has_segment_properties:
        output["segment_properties"] = "segment_properties"
    info.write_text(json.dumps(output, indent=2))

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


def determine_chunk_size_for_lod(
    mesh_shape: tuple[int, int, int] | np.ndarray,  # type: ignore
    max_lod: int,
    min_chunk_dim: int = 16,
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], int]:
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
    tuple[tuple[int, int, int], tuple[int, int, int]], int
        The actual chunk size, the min chunk size, and the number of LODs that can be generated
    """
    mesh_shape: np.ndarray = np.array(mesh_shape)  # type: ignore

    def _determine_chunk_shape_for_lod(mesh_shape, lod_level):
        return 2 ** np.floor(np.log2(mesh_shape / 2**lod_level))

    # Find a power of 2 chunk size to reach the max LOD
    chunk_shape = _determine_chunk_shape_for_lod(mesh_shape, max_lod)

    # If the chunk size is smaller than the minimum chunk dimension
    # then we can't respect the max LOD and need to reduce the LODs
    if np.any(chunk_shape < min_chunk_dim):
        max_lod = int(max(np.min(np.log2(mesh_shape / min_chunk_dim)), 0))
        chunk_shape = _determine_chunk_shape_for_lod(mesh_shape, max_lod)
    final_lod = int(max(np.min(np.log2(mesh_shape / chunk_shape)), 0)) + 1
    LOGGER.info(
        "Will produce %i LODs for this mesh at min size %s",
        final_lod,
        chunk_shape,
    )
    x, y, z = chunk_shape.astype(int)

    # Compute the Actual Chunk Size (ACS)
    acs_x, acs_y, acs_z = np.ceil(mesh_shape / 2 ** (final_lod - 1))

    return (((int(acs_x), int(acs_y), int(acs_z)), (int(x), int(y), int(z))), final_lod)


def clean_mesh_folder(output_path: str | Path, mesh_directory: str = "mesh"):
    """Remove unnecessary files from the mesh folder

    The conversion produces extra unnecessary files, so we clean up
    The only needed files are the info file, and any file that ends with .shard
    However, if the sharding fails, we may have other files, so we only delete
    If there is at least one shard file
    """
    output_path = Path(output_path)
    mesh_dir = output_path / mesh_directory
    if mesh_dir.exists() and any(f.name.endswith(".shard") for f in mesh_dir.iterdir()):
        LOGGER.debug("Cleaning up the mesh directory")
        for f in mesh_dir.iterdir():
            if f.is_file() and not (f.name == "info" or f.name.endswith(".shard")):
                f.unlink()
    else:
        LOGGER.warning("No shard files found, not cleaning up the mesh directory")


def generate_mesh_from_lods(
    lods: list[trimesh.Scene],
    outfolder: str | Path,
    min_mesh_chunk_dim: int = 16,
    label: int = 1,
    bounding_box_size: tuple[float, float, float] | None = None,
    string_label: str | None = None,
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
    string_label : str | None, optional
        The string label to use, by default None - ie. not used
    """
    concatenated_lods: list[trimesh.Trimesh] = cast(list[trimesh.Trimesh], [lod.to_geometry() for lod in lods])
    num_lod = len(concatenated_lods)
    first_lod = 0
    _, bb2 = concatenated_lods[first_lod].bounds

    def _compute_size(bbx):
        max_bound = np.ceil(bbx)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = bounding_box_size if bounding_box_size is not None else _compute_size(bb2)

    _, mesh_shape = determine_mesh_shape_from_lods(concatenated_lods)
    (actual_chunk_shape, smallest_chunk_size), calculated_num_lod = determine_chunk_size_for_lod(
        mesh_shape,
        num_lod - 1,
        min_chunk_dim=min_mesh_chunk_dim,
    )

    # The resolution is not handled here, but in the neuroglancer state
    _generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        mesh_chunk_size=actual_chunk_shape,
        has_segment_properties=string_label is not None,
    )

    if string_label is not None:
        write_segment_properties(outfolder, [label], [string_label])

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
    string_label: str | None = None,
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
    generate_multilabel_multiresolution_mesh(
        outfolder=outfolder,
        max_lod=max_lod,
        min_mesh_chunk_dim=min_mesh_chunk_dim,
        bounding_box_size=bounding_box_size,
        label_dict={label: glb},
        string_labels={label: string_label} if string_label is not None else None,
    )


def generate_multilabel_multiresolution_mesh(
    label_dict: dict[int, trimesh.Scene | str | Path],
    outfolder: str | Path,
    max_lod: int = 2,
    min_mesh_chunk_dim: int = 16,
    bounding_box_size: tuple[float, float, float] | None = None,
    string_labels: dict[int, str] | None = None,
):
    """
    Generate standalone sharded multiresolution meshes from a mapping of labels to glb files or scenes.

    The multilabel version allows for multiple mesh labels to be stored in the same
    segmentation file. This can be useful if multiple meshes are desired to be stored
    but all of them are part of the same segmentation - so they are in one neuroglancer
    layer.

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
        mesh: trimesh.Trimesh = scene.to_geometry()  # type: ignore
        labels[k] = mesh

    mesh = next(iter(labels.values()))
    _, bb2 = mesh.bounds  # type: ignore

    def _compute_size():
        max_bound = np.ceil(bb2)
        return np.maximum(max_bound, np.full(3, 1))

    size_x, size_y, size_z = bounding_box_size if bounding_box_size is not None else _compute_size()

    # get the biggest smallest_chunk_size for all mesh
    # we take the lowest possible value to setup the max search
    smallest_chunk_sizes = (((0, 0, 0), (0, 0, 0)), 0)
    for mesh in labels.values():
        mesh_shape = _determine_mesh_shape(mesh)
        # (actual_chunk_size, smallest_chunk_size), computed_num_lod is returned
        chunk_sizes, computed_num_lod = determine_chunk_size_for_lod(
            mesh_shape,
            max_lod,
            min_mesh_chunk_dim,
        )
        is_larger = chunk_sizes[1] > smallest_chunk_sizes[0][1]
        LOGGER.debug(
            "Comparing new %s with old %s for larger chunk. New chunk is larger? %s",
            chunk_sizes[1],
            smallest_chunk_sizes[0][1],
            is_larger,
        )
        if is_larger:
            smallest_chunk_sizes = (chunk_sizes, computed_num_lod)
    (actual_chunk_size, smallest_chunk_size), computed_num_lod = smallest_chunk_sizes

    # The resolution is not handled here, but in the neuroglancer state
    _generate_standalone_mesh_info(
        outfolder,
        size=(size_x, size_y, size_z),
        mesh_chunk_size=actual_chunk_size,
        has_segment_properties=string_labels is not None,
    )

    if string_labels is not None:
        ids = list(string_labels.keys())
        string_labels_list = [string_labels[i] for i in ids]
        write_segment_properties(outfolder, ids, string_labels_list)

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


def generate_multiresolution_mesh_from_segmentation(
    precomputed_segmentation_path: Path,
    mesh_directory: str,
    max_lod: int,
    mesh_shape: tuple[int, int, int] | np.ndarray,
    min_mesh_chunk_dim: int = 16,
    max_simplification_error: int = 10,
    labels_dict: dict[int, str] | None = None,
    fill_missing: bool = False,
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
    max_simplification_error: int
        The maximal simplification error allowed for the mesh generation from the
        segmentation. This is used to simplify the mesh and reduce the number of
        vertices and faces in the mesh. The error is in the same unit as the data.
    labels_dict: dict[int, str] | None
        A dictionary of labels to string labels. This is used to generate the segment properties
        for the segmentation. If None, no segment properties are generated.
    fill_missing: bool
        Fills in empty volumes with a volume filled with the label 0.
        By default, this is False. But can be set to True if the mesh bounds
        are determined via fast bounding box calculation.
    """
    tq = LocalTaskQueue()

    path = f"precomputed://file://{precomputed_segmentation_path}"
    LOGGER.info("Generating mesh for %s", path)
    LOGGER.debug("Meshing with zmesher and %i max simplification error", max_simplification_error)
    simplification = max_simplification_error != 0
    tasks = tc.create_meshing_tasks(
        path,
        mip=0,
        shape=(256, 256, 256),
        mesh_dir=mesh_directory,
        sharded=True,
        fill_missing=fill_missing,
        max_simplification_error=max_simplification_error,
        simplification=simplification,
    )
    tq.insert(tasks)
    tq.execute()

    LOGGER.debug("Creating mesh manifest")
    tasks = tc.create_mesh_manifest_tasks(path, mesh_dir=mesh_directory, magnitude=3)
    tq.insert(tasks)
    tq.execute()

    LOGGER.debug("Creating multi-resolution mesh with max %i LOD", max_lod)
    (_, min_chunk_size), num_lods = determine_chunk_size_for_lod(
        mesh_shape,
        max_lod,
        min_mesh_chunk_dim,
    )
    LOGGER.info("Generating %i LODs for the mesh", num_lods)
    tasks = tc.create_sharded_multires_mesh_tasks(
        path,
        mesh_dir=mesh_directory,
        num_lod=max_lod,
        min_chunk_size=min_chunk_size,
    )
    tq.insert(tasks)
    tq.execute()

    if labels_dict is not None:
        LOGGER.debug("Writing segment properties")
        write_segment_properties(
            Path(mesh_directory).parent,
            list(labels_dict.keys()),
            list(labels_dict.values()),
        )


def generate_single_resolution_mesh(
    dask_data: da.Array,
    output_path: Path,
    mesh_directory: str,
    resolution: tuple[float, float, float],
) -> None:
    """
    ! This function is deprecated, please use "generate_multiresolution_mesh_from_segmentation" instead !
    Create a single res mesh for the given volume if a mesh directory is provided

    This uses the neuroglancer local volume to create the mesh. This can be a challenge
    for large volumes, as the mesh is created in memory. For this reason, it is recommended to instead use the multi-resolution mesh generation, but set the number of LODs to 1.
    This method is useful for small volumes or for testing purposes, or to compare the mesh generation with the multi-resolution mesh generation.
    """
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
    LOGGER.info("Wrote segmentation mesh to %s", mesh_path)
