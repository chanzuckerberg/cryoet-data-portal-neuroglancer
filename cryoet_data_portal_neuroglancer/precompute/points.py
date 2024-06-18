import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from neuroglancer import AnnotationPropertySpec, CoordinateSpace
from neuroglancer.write_annotations import AnnotationWriter

from cryoet_data_portal_neuroglancer.sharding import ShardingSpecification, jsonify
from cryoet_data_portal_neuroglancer.utils import rotate_xyz_via_matrix


def _line_index_to_rgb(line_index: int) -> tuple[int, int, int]:
    """x = red, y = green, z = blue"""
    line_to_rgb = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
    }
    return line_to_rgb.get(line_index, (255, 255, 255))


def _write_annotations_oriented(
    output_dir: Path,
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    coordinate_space: CoordinateSpace,
    names_by_id: dict[int, str],
    label_key_mapper: Callable[[dict[str, Any]], int],
    color_mapper: Callable[[dict[str, Any]], tuple[int, int, int]],
) -> Path:
    """
    Create a neuroglancer annotation folder with the given annotations.
    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    writer = AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="line",
        properties=[
            AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=list(names_by_id.keys()),
                enum_labels=list(names_by_id.values()),
            ),
            AnnotationPropertySpec(id="diameter", type="float32"),
            AnnotationPropertySpec(id="point_index", type="float32"),
            AnnotationPropertySpec(id="point_color", type="rgb"),
            AnnotationPropertySpec(id="line_color", type="rgb"),
        ],
    )

    # Using 10nm as default size
    diameter = metadata["annotation_object"].get("diameter", 100) / 10
    # Make the line length be a little longer than the diameter
    # This can't be changed in post, and has to be done at the time of encoding
    line_distance = diameter * 1.5
    for index, p in enumerate(data):
        rotated_xyz = rotate_xyz_via_matrix(p["xyz_rotation_matrix"])
        start_point = np.array([p["location"][k] for k in ("x", "y", "z")])
        for i in range(3):
            end_point = start_point + line_distance * rotated_xyz[i]
            if not np.isclose(np.linalg.norm(end_point - start_point), line_distance):
                raise ValueError(
                    "Incorrect input rotation matrix, resulting in incorrect line length for oriented points.",
                )
            writer.add_line(
                start_point,
                end_point,
                diameter=diameter,
                point_index=float(index),
                name=label_key_mapper(p),
                point_color=color_mapper(p),
                line_color=_line_index_to_rgb(i),
            )
    writer.properties.sort(key=lambda prop: prop.id != "name")
    writer.write(output_dir)
    return output_dir


def _write_annotations(
    output_dir: Path,
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    coordinate_space: CoordinateSpace,
    names_by_id: dict[int, str],
    label_key_mapper: Callable[[dict[str, Any]], int],
    color_mapper: Callable[[dict[str, Any]], tuple[int, int, int]],
) -> Path:
    """
    Create a neuroglancer annotation folder with the given annotations.
    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    writer = AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=list(names_by_id.keys()),
                enum_labels=list(names_by_id.values()),
            ),
            AnnotationPropertySpec(id="diameter", type="float32"),
            AnnotationPropertySpec(id="point_index", type="float32"),
            AnnotationPropertySpec(id="color", type="rgb"),
        ],
    )

    # Using 10nm as default size
    diameter = metadata["annotation_object"].get("diameter", 100) / 10
    for index, p in enumerate(data):
        location = [p["location"][k] for k in ("x", "y", "z")]
        writer.add_point(
            location,
            diameter=diameter,
            point_index=float(index),
            name=label_key_mapper(p),
            color=color_mapper(p),
        )
    writer.properties.sort(key=lambda prop: prop.id != "name")
    writer.write(output_dir)
    return output_dir


def _shard_by_id_index(directory: Path, shard_bits: int, minishard_bits: int):
    sharding_specification = ShardingSpecification(
        type="neuroglancer_uint64_sharded_v1",
        preshift_bits=0,
        hash="identity",
        minishard_bits=minishard_bits,
        shard_bits=shard_bits,
        minishard_index_encoding="gzip",
        data_encoding="gzip",
    )
    labels = {}
    for file in (directory / "by_id").iterdir():
        if ".shard" not in file.name:
            labels[int(file.name)] = file.read_bytes()
            file.unlink()

    shard_files = sharding_specification.synthesize_shards(labels, progress=True)
    for shard_filename, shard_content in shard_files.items():
        (directory / "by_id" / shard_filename).write_bytes(shard_content)

    info_path = directory / "info"
    info = json.load(info_path.open("r", encoding="utf-8"))
    info["by_id"]["sharding"] = sharding_specification.to_dict()
    info_path.write_text(jsonify(info, indent=2))


def encode_annotation(
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: Path,
    resolution: float,
    is_oriented: bool = False,
    names_by_id: dict[int, str] = None,
    label_key_mapper: Callable[[dict[str, Any]], int] = lambda x: 0,
    color_mapper: Callable[[dict[str, Any]], tuple[int, int, int]] = lambda x: (255, 255, 255),
    shard_by_id: tuple[int, int] = (0, 10),
) -> None:
    if shard_by_id and len(shard_by_id) < 2:
        shard_by_id = (0, 10)
    coordinate_space = CoordinateSpace(
        names=["x", "y", "z"],
        units=["m", "m", "m"],
        scales=[resolution, resolution, resolution],
    )
    if names_by_id is None:
        names_by_id = {0: metadata.get("annotation_object", {}).get("name", "")}
    writer_function = _write_annotations_oriented if is_oriented else _write_annotations
    writer_function(
        output_path,
        data,
        metadata,
        coordinate_space,
        names_by_id,
        label_key_mapper,
        color_mapper,
    )
    print("Wrote annotations to", output_path)

    if shard_by_id and len(shard_by_id) == 2:
        shard_bits, minishard_bits = shard_by_id
        _shard_by_id_index(output_path, shard_bits, minishard_bits)
