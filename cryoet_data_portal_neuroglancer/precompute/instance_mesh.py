import logging
import typing
from typing import Any

import numpy as np
import trimesh
from tqdm import tqdm

from cryoet_data_portal_neuroglancer.precompute.mesh import decimate_mesh
from cryoet_data_portal_neuroglancer.utils import rotate_and_translate_mesh

LOGGER = logging.getLogger(__name__)


def encode_oriented_mesh(
    input_geometry: trimesh.Scene | trimesh.Trimesh,
    data: list[dict[str, Any]],
    num_lods: int = 3,
    max_faces_for_first_lod: int = 5_000_000,
    decimation_aggressiveness: float = 4.5,
) -> list[trimesh.Scene]:
    """Turn a mesh into an oriented mesh with a list of orientations and translations

    Parameters
    ----------
    input_geometry : trimesh.Scene | trimesh.Trimesh
        The scene containing the mesh or the mesh itself
    data : list[dict[str, Any]]
        The list of orientations and translations
    num_lods: int, optional
        The number of levels of detail to generate, by default 3
        A high resolution, a medium resolution, and a low resolution
    max_faces : int, optional
        The maximum number of faces per mesh, by default 5million
        This determines the first LOD that is used.
        For example, if LOD 0 has 6 million faces when copied to all the positions,
        and LOD 1 has 3 million faces when copied to all the positions,
        and the maximum faces is 5 million, then LOD 1 is used as the first LOD.
        The remaining LODs are then generated starting from LOD 1
    decimation_aggressiveness : float, optional
        The aggressiveness of the decimation algorithm, by default 5.0

    Returns
    -------
    list[trimesh.Scene]
        The list of scenes containing the oriented meshes
    """
    scaled, decimated_meshes = scale_and_decimate_mesh(input_geometry, max(10, num_lods), decimation_aggressiveness)
    num_faces_per_lod = np.array([len(mesh.faces) for mesh in decimated_meshes])
    total_number_of_points = len(data)
    total_faces_per_lod = num_faces_per_lod * total_number_of_points
    if np.all(total_faces_per_lod > max_faces_for_first_lod):
        raise ValueError(
            f"Total faces per LOD {total_faces_per_lod} are all greater than the maximum faces {max_faces_for_first_lod}.",
        )
    first_lod = np.argmax(total_faces_per_lod <= max_faces_for_first_lod)
    # Now we have the first LOD that is less than the maximum faces, and redo the decimation to ensure that we have the correct number of LODs
    num_total_lods = first_lod + num_lods
    decimated_meshes = decimate_mesh(
        scaled,
        num_total_lods,
        as_trimesh=True,
        aggressiveness=decimation_aggressiveness,
    )
    LOGGER.info(
        "Using LOD %i as the first LOD, with %i faces, which is less than %i maximum faces. There are %i LODs remaining in total.",
        first_lod,
        total_faces_per_lod[first_lod],
        max_faces_for_first_lod,
        len(decimated_meshes) - first_lod,
    )

    results = []
    for mesh in tqdm(decimated_meshes[first_lod:], desc="Processing meshes into LODs and positions"):  # type: ignore
        new_scene = trimesh.Scene()
        for index, point in enumerate(data):
            translation = np.array([point["location"][k] for k in ("x", "y", "z")])
            rotation = np.array(point["xyz_rotation_matrix"])
            rotate_and_translate_mesh(mesh, new_scene, index, rotation, translation)
        results.append(new_scene)

    return results


def scale_and_decimate_mesh(
    input_geometry: trimesh.Scene | trimesh.Trimesh,
    num_lods: int,
    decimation_aggressiveness: float = 4.5,
) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    if isinstance(input_geometry, trimesh.Trimesh):
        mesh = input_geometry
    else:
        geometry = input_geometry.geometry
        if len(geometry) > 1:
            raise ValueError("Scene has more than one mesh")
        mesh: trimesh.Trimesh = next(v for v in geometry.values())
    # The co-ordinate system of the mesh is in angstrom
    # As such, one unit of the mesh is 0.1 nm
    mesh_resolution = 0.1  # nm
    # Since meshes are in angstrom, we need to scale it to nanometers
    scaled = typing.cast(trimesh.Trimesh, mesh.copy().apply_scale(mesh_resolution))
    # We don't need to scale to the real tomogram resolution, because we make
    # the hard assumption that the resolution of the output mesh
    # is 1.0 nm, and then we scale the mesh to the real resolution
    # of the tomogram inside of the neuroglancer viewer
    # instead of at the time of encoding the mesh

    decimated_meshes = decimate_mesh(
        scaled,
        num_lods,
        aggressiveness=decimation_aggressiveness,
        as_trimesh=True,
    )

    return scaled, decimated_meshes
