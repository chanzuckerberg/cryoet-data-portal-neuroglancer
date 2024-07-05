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
    resolution: float,
):
    geometry = scene.geometry
    if len(geometry) > 1:
        raise ValueError("Scene has more than one mesh")
    mesh = next(v for v in geometry.values())
    new_scene = trimesh.Scene()
    for index, point in tqdm.tqdm(
        enumerate(data),
        total=len(data),
        desc="Rotating and Translating Instanced Mesh",
    ):
        translation = np.array([point["location"][k] for k in ("x", "y", "z")])
        rotation = np.array(point["xyz_rotation_matrix"])
        rotate_and_translate_mesh(mesh, new_scene, index, rotation, translation)

    new_scene.export(output_path / "glb_mesh.glb", file_type="glb")
    return new_scene
