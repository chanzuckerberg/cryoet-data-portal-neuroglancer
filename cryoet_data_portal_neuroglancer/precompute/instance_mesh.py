from typing import Any

import numpy as np
import trimesh
from tqdm import tqdm

from cryoet_data_portal_neuroglancer.precompute.glb_meshes import decimate_mesh
from cryoet_data_portal_neuroglancer.utils import rotate_and_translate_mesh


def encode_oriented_mesh(
    input_geometry: trimesh.Scene | trimesh.Trimesh,
    data: list[dict[str, Any]],
    num_lods: int = 1,
) -> list[trimesh.Scene]:
    """Turn a mesh into an oriented mesh with a list of orientations and translations

    Parameters
    ----------
    input_geometry : trimesh.Scene | trimesh.Trimesh
        The scene containing the mesh or the mesh itself
    data : list[dict[str, Any]]
        The list of orientations and translations
    num_lods : int, optional
        The number of levels of detail to generate, by default 1

    Returns
    -------
    list[trimesh.Scene]
        The list of scenes containing the oriented meshes
    """
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
    scaled = mesh.copy().apply_scale(mesh_resolution)
    # We don't need to scale to the real tomogram resolution, because we make
    # the hard assumption that the resolution of the output mesh
    # is 1.0 nm, and then we scale the mesh to the real resolution
    # of the tomogram inside of the neuroglancer viewer
    # instead of at the time of encoding the mesh

    decimated_meshes = decimate_mesh(scaled, num_lods, as_trimesh=True)

    results = []
    for mesh in tqdm(decimated_meshes, desc="Processing meshes into LODs and positions"):
        new_scene = trimesh.Scene()
        for index, point in enumerate(data):
            translation = np.array([point["location"][k] for k in ("x", "y", "z")])
            rotation = np.array(point["xyz_rotation_matrix"])
            rotate_and_translate_mesh(mesh, new_scene, index, rotation, translation)
        results.append(new_scene)

    return results
