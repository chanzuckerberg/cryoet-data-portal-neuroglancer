from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import trimesh

from cryoet_data_portal_neuroglancer.utils import rotate_and_translate_mesh


def encode_oriented_mesh(
    mesh: "trimesh.Trimesh",
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: Path,
    resolution: float,
):
    meshes = []
    for index, point in tqdm.tqdm(
        enumerate(data),
        total=len(data),
        desc="Rotating and Translating Instanced Mesh",
    ):
        translation = np.array([point["location"][k] for k in ("x", "y", "z")])
        rotation = np.array(point["xyz_rotation_matrix"])
        rotated_mesh = rotate_and_translate_mesh(mesh, rotation, translation)
        meshes.append(rotated_mesh)

    combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh.export(output_path / "glb_mesh.glb", file_type="glb")
