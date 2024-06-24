from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from neuroglancer import AnnotationPropertySpec, CoordinateSpace
from neuroglancer.write_annotations import AnnotationWriter

from cryoet_data_portal_neuroglancer.models.annotation_encoder import AnnotationEncoder
from cryoet_data_portal_neuroglancer.utils import rotate_xyz_via_matrix


class LineLengthMethod(Enum):
    """Method for determining line length for oriented points."""

    SCALE = auto()
    ABSOLUTE = auto()

    def __str__(self):
        return self.name.lower()


@dataclass
class OrientedPointAnnotationEncoder(AnnotationEncoder):
    line_width_value: float = 1.5
    line_width_method: LineLengthMethod = LineLengthMethod.SCALE

    def _prepare_writer_specifications(self):
        self._writer = AnnotationWriter(
            coordinate_space=self.coordinate_space,
            annotation_type="line",
            properties=[
                AnnotationPropertySpec(
                    id="name",
                    type="uint8",
                    enum_values=list(self.names_by_id.keys()),
                    enum_labels=list(self.names_by_id.values()),
                ),
                AnnotationPropertySpec(id="diameter", type="float32"),
                AnnotationPropertySpec(id="point_index", type="float32"),
                AnnotationPropertySpec(id="point_color", type="rgb"),
                AnnotationPropertySpec(id="line_color", type="rgb"),
            ],
        )

    def process_data_to_writer_specifications(self, data: list[dict[str, Any]], metadata: dict[str, Any]):
        # Using 10nm as default size
        diameter = metadata["annotation_object"].get("diameter", 100) / 10
        if self.line_width_method == LineLengthMethod.SCALE:
            line_distance = diameter * self.line_width_value
        else:
            line_distance = self.line_width_value
        for index, p in enumerate(data):
            rotated_xyz = rotate_xyz_via_matrix(p["xyz_rotation_matrix"])
            start_point = np.array([p["location"][k] for k in ("x", "y", "z")])
            for i in range(3):
                end_point = start_point + line_distance * rotated_xyz[i]
                if not np.isclose(np.linalg.norm(end_point - start_point), line_distance):
                    raise ValueError(
                        "Incorrect input rotation matrix, resulting in incorrect line length for oriented points.",
                    )
                self._writer.add_line(
                    start_point,
                    end_point,
                    diameter=diameter,
                    point_index=float(index),
                    name=self.label_key_mapper(p),
                    point_color=self.color_mapper(p),
                    line_color=self._line_index_to_rgb(i),
                )

    @staticmethod
    def _line_index_to_rgb(line_index: int) -> tuple[int, int, int]:
        """x = red, y = green, z = blue"""
        line_to_rgb = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
        }
        return line_to_rgb.get(line_index, (255, 255, 255))


@dataclass
class PointAnnotationEncoder(AnnotationEncoder):
    def _prepare_writer_specifications(self):
        self._writer = AnnotationWriter(
            coordinate_space=self.coordinate_space,
            annotation_type="point",
            properties=[
                AnnotationPropertySpec(
                    id="name",
                    type="uint8",
                    enum_values=list(self.names_by_id.keys()),
                    enum_labels=list(self.names_by_id.values()),
                ),
                AnnotationPropertySpec(id="diameter", type="float32"),
                AnnotationPropertySpec(id="point_index", type="float32"),
                AnnotationPropertySpec(id="color", type="rgb"),
            ],
        )

    def process_data_to_writer_specifications(self, data: list[dict[str, Any]], metadata: dict[str, Any]):
        # Using 10nm as default size
        diameter = metadata["annotation_object"].get("diameter", 100) / 10
        for index, p in enumerate(data):
            location = [p["location"][k] for k in ("x", "y", "z")]
            self._writer.add_point(
                location,
                diameter=diameter,
                point_index=float(index),
                name=self.label_key_mapper(p),
                color=self.color_mapper(p),
            )


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
    oriented_line_parameters: tuple[float, LineLengthMethod] = (1.5, LineLengthMethod.SCALE),
) -> None:
    """Encode annotation data to Neuroglancer format.

    Parameters
    ----------
    data : list[dict[str, Any]]
        List of dictionaries containing annotation data from ndjson file.
    metadata : dict[str, Any]
        Metadata for the annotation data from json file.
    output_path : Path
        Path to the output directory for the encoded annotations.
    resolution : float
        Resolution of the data in nanometers. Assumed the same in all 3 dimensions.
    is_oriented : bool, optional
        Whether the annotations are oriented points, by default False
    names_by_id : dict[int, str], optional
        Mapping of annotation id to name, by default None
    label_key_mapper : Callable[[dict[str, Any]], int], optional
        Function to map annotation data to label key, by default, all annotations are labeled as 0
    color_mapper : Callable[[dict[str, Any]], tuple[int, int, int]], optional
        Function to map annotation data to RGB color, by default all annotations are white
    shard_by_id : Optional[tuple[int, int]], optional
        Tuple of shard_bits and minishard_bits, by default (0, 10)
        If None, no sharding will be done.
    line_length_parameters : tuple[float, LineLengthMethod], optional
        Tuple of line length and method for determining line length for oriented points, by default (1.5, LineLengthMethod.SCALE)
    """
    if shard_by_id and len(shard_by_id) < 2:
        shard_by_id = (0, 10)
    coordinate_space = CoordinateSpace(
        names=["x", "y", "z"],
        units=["m", "m", "m"],
        scales=[resolution, resolution, resolution],
    )
    if names_by_id is None:
        names_by_id = {0: metadata.get("annotation_object", {}).get("name", "")}

    annotation_encoder_type = OrientedPointAnnotationEncoder if is_oriented else PointAnnotationEncoder
    kwargs = (
        {}
        if not is_oriented
        else {"line_width_value": oriented_line_parameters[0], "line_width_method": oriented_line_parameters[1]}
    )
    annotation_encoder = annotation_encoder_type(
        output_path=output_path,
        resolution=resolution,
        coordinate_space=coordinate_space,
        shard_by_id=shard_by_id,
        names_by_id=names_by_id,
        label_key_mapper=label_key_mapper,
        color_mapper=color_mapper,
        **kwargs,
    )
    annotation_encoder.process_data_to_writer_specifications(data, metadata)
    annotation_encoder.write_annotations()
