from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np


def make_transform(input_dict: dict, dim: str, resolution: float):
    input_dict[dim] = [resolution, "m"]


class RenderingTypes(Enum):
    """Types of rendering for Neuroglancer."""

    SEGMENTATION = auto()
    IMAGE = auto()
    ANNOTATION = auto()

    def __str__(self):
        return self.name.lower()


@dataclass
class RenderingJSONGenerator:
    """Generates a JSON file for Neuroglancer to read."""

    source: str
    name: str

    @property
    def layer_type(self) -> str:
        """Returns the layer type for Neuroglancer."""
        try:
            return str(self._type)  # type: ignore
        except AttributeError:
            raise ValueError(f"Unknown rendering type {self._type}") from None  # type: ignore

    def to_json(self) -> dict:
        return self.generate_json()

    @abstractmethod
    def generate_json(self) -> dict:
        """Generates the JSON for Neuroglancer."""
        raise NotImplementedError


@dataclass
class ImageJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    resolution: tuple[float, float, float]
    size: dict[str, float]
    contrast_limits: tuple[float, float] = (-64, 64)
    start: dict[str, float] = None
    mean: float = None
    rms: float = None

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def _create_shader_and_controls(self) -> dict[str, Any]:
        if self.mean is None or self.rms is None:
            distance = self.contrast_limits[1] - self.contrast_limits[0]
            window_start = self.contrast_limits[0] - (distance / 10)
            window_end = self.contrast_limits[1] + (distance / 10)
            shader = (
                f"#uicontrol invlerp contrast(range=[{self.contrast_limits[0]}, {self.contrast_limits[1]}], "
                f"window=[{window_start}, {window_end}])\nvoid main() {{\n  emitGrayscale(contrast());\n}}"
            )
            return {"shader": shader}

        width = 3 * self.rms
        start = self.mean - width
        end = self.mean + width
        window_width_factor = width * 0.1
        window_start = start - window_width_factor
        window_end = end + window_width_factor
        return {
            "shader": "#uicontrol invlerp normalized\n\nvoid main() {\n  emitGrayscale(normalized());\n}\n",
            "shaderControls": {
                "normalized": {
                    "range": [start, end],
                    "window": [window_start, window_end],
                },
            },
        }

    def _get_computed_values(self) -> dict[str, Any]:
        nstart = self.start or {k: 0 for k in "xyz"}
        avg_cross_section_render_height = 400
        largest_dimension = max([self.size.get(d, 0) - nstart.get(d, 0) for d in "xyz"])
        return {
            "_position": [np.round(np.mean([self.size.get(d, 0), nstart.get(d, 0)])) for d in "xyz"],
            "_crossSectionScale": max(largest_dimension / avg_cross_section_render_height, 1),
        }

    def generate_json(self) -> dict:
        transform: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1], strict=False):
            make_transform(transform, dim, resolution)

        original: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1], strict=False):
            make_transform(original, dim, resolution)

        config = {
            "type": self.layer_type,
            "name": self.name,
            "source": {
                "url": f"zarr://{self.source}",
                "transform": {
                    "outputDimensions": transform,
                    "inputDimensions": original,
                },
            },
            "opacity": 0.51,
            "tab": "rendering",
        }
        return {**config, **self._create_shader_and_controls(), **self._get_computed_values()}


@dataclass
class AnnotationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: str
    point_size_multiplier: float = 1.0

    def __post_init__(self):
        self._type = RenderingTypes.ANNOTATION

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": f"precomputed://{self.source}",
            "tab": "rendering",
            "shader": self._get_shader(),
        }

    def _get_shader(self):
        return (
            f"#uicontrol float pointScale slider(min=0.01, max=2.0, default={self.point_size_multiplier}, step=0.01)\n"
            f"void main() {{\n"
            f"  setColor({self.color});\n"
            f"  setPointMarkerSize(pointScale * prop_diameter());\n"
            f"  setPointMarkerBorderWidth(0.1);\n"
            f"}}"
        )


@dataclass
class SegmentationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: tuple[str, str]

    def __post_init__(self):
        self._type = RenderingTypes.SEGMENTATION

    def generate_json(self) -> dict:
        color_part = f" ({self.color[1]})" if self.color[1] else ""
        return {
            "type": self.layer_type,
            "name": f"{self.name}{color_part}",
            "source": f"precomputed://{self.source}",
            "tab": "rendering",
            "selectedAlpha": 1,
            "hoverHighlight": False,
            "segments": [
                1,
            ],
            "segmentDefaultColor": self.color[0],
        }
