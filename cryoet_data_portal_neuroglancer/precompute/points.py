import json
from pathlib import Path
from typing import Any, Callable, Optional

from neuroglancer import AnnotationPropertySpec, CoordinateSpace
from neuroglancer.write_annotations import AnnotationWriter

from cryoet_data_portal_neuroglancer.sharding import ShardingSpecification, jsonify


def _build_rotation_matrix_properties() -> list[AnnotationPropertySpec]:
    return [AnnotationPropertySpec(id=f"rot_mat_{i}_{j}", type="float32") for i in range(3) for j in range(3)]


def _write_annotations(
    output_dir: Path,
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    coordinate_space: CoordinateSpace,
    is_oriented: bool,
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
                type="uint8" if len(names_by_id) < 2**8 else "uint16",
                enum_values=list(names_by_id.keys()),
                enum_labels=list(names_by_id.values()),
            ),
            AnnotationPropertySpec(id="diameter", type="float32"),
            AnnotationPropertySpec(id="point_index", type="float32"),
            AnnotationPropertySpec(id="color", type="rgb"),
            # Spec must be added at the object construction time, not after
            *(_build_rotation_matrix_properties() if is_oriented else []),
        ],
    )

    # Using 10nm as default size
    diameter = metadata["annotation_object"].get("diameter", 100) / 10
    for index, p in enumerate(data):
        location = [p["location"][k] for k in ("x", "y", "z")]
        rot_mat = {}
        if is_oriented:
            rot_mat = {
                f"rot_mat_{i}_{j}": col for i, line in enumerate(p["xyz_rotation_matrix"]) for j, col in enumerate(line)
            }
        writer.add_point(
            location,
            diameter=diameter,
            point_index=float(index),
            name=label_key_mapper(p),
            color=color_mapper(p),
            **rot_mat,
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
    shard_by_id: Optional[tuple[int, int]] = (0, 10),
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
    _write_annotations(
        output_path,
        data,
        metadata,
        coordinate_space,
        is_oriented,
        names_by_id,
        label_key_mapper,
        color_mapper,
    )
    print("Wrote annotations to", output_path)

    if shard_by_id and len(shard_by_id) == 2:
        shard_bits, minishard_bits = shard_by_id
        _shard_by_id_index(output_path, shard_bits, minishard_bits)
