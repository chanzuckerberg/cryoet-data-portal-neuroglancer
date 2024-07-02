from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from cryoet_data_portal_neuroglancer.utils import rotate_and_translate_mesh


def encode_oriented_mesh(
    mesh: "trimesh.Trimesh",
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: Path,
    resolution: float,
):
    for index, point in enumerate(data):
        translation = np.array([point["location"][k] for k in ("x", "y", "z")])
        rotation = point["xyz_rotation_matrix"]
        rotated_mesh = rotate_and_translate_mesh(mesh, rotation, translation)

    return rotated_mesh
