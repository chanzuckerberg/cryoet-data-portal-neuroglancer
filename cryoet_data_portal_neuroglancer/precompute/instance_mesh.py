from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import trimesh

from cryoet_data_portal_neuroglancer.utils import rotate_and_translate_mesh


def encode_oriented_mesh(
    scene: "trimesh.Scene",
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: Path,
    real_resolution: float,
):
    """Turn a mesh into an oriented mesh with a list of orientations and translations

    Parameters
    ----------
    scene : trimesh.Scene
        The scene containing the mesh
    data : list[dict[str, Any]]
        The list of orientations and translations
    metadata : dict[str, Any]
        The metadata for the oriented points
    output_path : Path
        The output path for the new mesh
    real_resolution : float
        The real resolution of the data units, or the voxel size.
        Must be in nanometers.
    """
    geometry = scene.geometry
    if len(geometry) > 1:
        raise ValueError("Scene has more than one mesh")
    mesh: trimesh.Trimesh = next(v for v in geometry.values())
    # The co-ordinate system of the mesh is in angstrom
    # As such, one unit of the mesh is 0.1 nm
    mesh_resolution = 0.1 # nm
    # Since meshes are in angstrom, we need to scale it to nanometers
    scaled = mesh.copy().apply_scale(mesh_resolution)
    # We don't need to scale to the real tomogram resolution, because we make
    # the hard assumption that the resolution of the output mesh
    # is 1.0 nm, and then we scale the mesh to the real resolution
    # of the tomogram inside of the neuroglancer viewer
    # instead of at the time of encoding the mesh
    new_scene = trimesh.Scene()
    for index, point in tqdm.tqdm(
        enumerate(data),
        total=len(data),
        desc="Rotating and Translating Instanced Mesh",
    ):
        translation = np.array([point["location"][k] for k in ("x", "y", "z")])
        rotation = np.array(point["xyz_rotation_matrix"])
        rotate_and_translate_mesh(scaled, new_scene, index, rotation, translation)

    output_path.mkdir(exist_ok=True, parents=True)
    new_scene.export(output_path / "glb_mesh.glb", file_type="glb")
    return new_scene
