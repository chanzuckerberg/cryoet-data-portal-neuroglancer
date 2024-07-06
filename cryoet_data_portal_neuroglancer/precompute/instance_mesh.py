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
    # Assuming the mesh resolution is in picometers
    mesh_resolution = scene.scale * 0.001
    print(f"Mesh resolution: {mesh_resolution}")
    resolution_ratio = mesh_resolution / real_resolution
    print(f"Resolution ratio: {resolution_ratio}")
    scaled = mesh.copy().apply_scale(resolution_ratio)
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
