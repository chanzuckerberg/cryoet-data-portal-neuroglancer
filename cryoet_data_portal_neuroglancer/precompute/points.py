import json
from pathlib import Path
from typing import Any, Optional

from neuroglancer import CoordinateSpace, AnnotationPropertySpec
from neuroglancer.write_annotations import AnnotationWriter

from cryoet_data_portal_neuroglancer.sharding import ShardingSpecification, jsonify
from cryoet_data_portal_neuroglancer.state.state_generator import setup_creation, process_color, AnnotationJSONGenerator


def build_rotation_matrix_properties() -> list[AnnotationPropertySpec]:
    return [
        AnnotationPropertySpec(id=f"rot_mat_{i}_{j}", type="float32")
        for i in range(3)
        for j in range(3)
    ]


def write_annotations(
    output_dir: Path,
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    coordinate_space: CoordinateSpace,
    is_oriented: bool,
) -> Path:
    """
    Create a neuroglancer annotation folder with the given annotations.

    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    name = metadata["annotation_object"]["name"]
    writer = AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=[0],
                enum_labels=[name],
            ),
            AnnotationPropertySpec(id="diameter", type="float32"),
            AnnotationPropertySpec(id="point_index", type="float32"),
            # Spec must be added at the object construction time, not after
            *(build_rotation_matrix_properties() if is_oriented else []),
        ],
    )
    # Convert angstrom to nanometer
    # Using 10nm as default size
    diameter = metadata["annotation_object"].get("diameter", 100) / 10
    for index, p in enumerate(data):
        location = [p["location"][k] for k in ("x", "y", "z")]
        if is_oriented:
            rot_mat = {
                f"rot_mat_{i}_{j}": col
                for i, line in enumerate(p["xyz_rotation_matrix"])
                for j, col in enumerate(line)
            }
        else:
            rot_mat = {}
        writer.add_point(
            location,
            diameter=diameter,
            point_index=float(index),
            name=0,
            **rot_mat,
        )
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


def precompute_point_data(
    data: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
    resolution: float,
    is_oriented: bool = False,
    shard_by_id: tuple[int, int] = (0, 10),
) -> None:
    if shard_by_id and len(shard_by_id) < 2:
        shard_by_id = (0, 10)

    coordinate_space = CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=[resolution, resolution, resolution],
    )
    write_annotations(output, data, metadata, coordinate_space, is_oriented)
    print("Wrote annotations to", output)

    if shard_by_id and len(shard_by_id) == 2:
        shard_bits, minishard_bits = shard_by_id
        _shard_by_id_index(output, shard_bits, minishard_bits)


def generate_state(
    source: str,
    name: Optional[str],
    color: str = None,
    point_size_multiplier: float = 1.0,
    oriented: bool = False,
) -> dict[str, Any]:
    source, name, url, _, _ = setup_creation(source, name, None, None, None)
    new_color = process_color(color)
    return AnnotationJSONGenerator(
        source=source,
        name=name,
        color=new_color,
        point_size_multiplier=point_size_multiplier,
        oriented=oriented,
    ).to_json()
