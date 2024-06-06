from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from cryoet_data_portal_neuroglancer.models.shader_builder import ImageVolumeShaderBuilder
from cryoet_data_portal_neuroglancer.utils import get_window_limits_from_contrast_limits


def create_source(
    url: str,
    input_dimension: tuple[float, float, float],
    output_dimension: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "url": url,
        "transform": {
            "outputDimensions": {
                "x": [output_dimension[0], "m"],
                "y": [output_dimension[1], "m"],
                "z": [output_dimension[2], "m"],
            },
            "inputDimensions": {
                "x": [input_dimension[0], "m"],
                "y": [input_dimension[1], "m"],
                "z": [input_dimension[2], "m"],
            },
        },
    }


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
    scale: tuple[float, float, float]

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
    """Generates JSON Neuroglancer config for Image volume."""

    size: dict[str, float]
    contrast_limits: tuple[float, float] = (-64, 64)
    start: dict[str, float] = None
    mean: float = None
    rms: float = None
    is_visible: bool = True

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def _create_shader_and_controls(self) -> dict[str, Any]:
        if self.mean is None or self.rms is None:
            window_limits = get_window_limits_from_contrast_limits(self.contrast_limits)
            shader = (
                f"#uicontrol invlerp contrast(range=[{self.contrast_limits[0]}, {self.contrast_limits[1]}], "
                f"window=[{window_limits[0]}, {window_limits[1]}])\nvoid main() {{\n  emitGrayscale(contrast());\n}}"
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
            "_projectionScale": largest_dimension * 1.1,
        }

    def generate_json(self) -> dict:
        config = {
            "type": self.layer_type,
            "name": self.name,
            "source": create_source(f"zarr://{self.source}", self.scale, self.scale),
            "opacity": 0.51,
            "tab": "rendering",
            "visible": self.is_visible,
        }
        return {**config, **self._create_shader_and_controls(), **self._get_computed_values()}


@dataclass
class AnnotationJSONGenerator(RenderingJSONGenerator):
    """Generates JSON Neuroglancer config for point annotation."""

    color: str
    point_size_multiplier: float = 1.0
    is_instance_segmentation: bool = False
    is_visible: bool = True

    def __post_init__(self):
        self._type = RenderingTypes.ANNOTATION

    def _get_shader(self):
        set_color = "color"
        ui_color_control = f'#uicontrol vec3 color color(default="{self.color}")\n'
        if self.is_instance_segmentation:
            set_color = "prop_color()"
            ui_color_control = ""
        return (
            f"#uicontrol float pointScale slider(min=0.01, max=2.0, default={self.point_size_multiplier}, step=0.01)\n"
            f"#uicontrol float opacity slider(min=0, max=1, default=1)\n"
            f"{ui_color_control}\n"
            f"void main() {{\n"
            f"  setColor(vec4({set_color}, opacity));\n"
            f"  setPointMarkerSize(pointScale * prop_diameter());\n"
            f"  setPointMarkerBorderWidth(0.1);\n"
            f"}}"
        )

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": create_source(f"precomputed://{self.source}", self.scale, self.scale),
            "tab": "rendering",
            "shader": self._get_shader(),
            "visible": self.is_visible,
        }


@dataclass
class SegmentationJSONGenerator(RenderingJSONGenerator):
    """Generates JSON Neuroglancer config for segmentation mask."""

    color: str
    is_visible: bool = True

    def __post_init__(self):
        self._type = RenderingTypes.SEGMENTATION

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": create_source(f"precomputed://{self.source}", self.scale, self.scale),
            "tab": "rendering",
            "selectedAlpha": 1,
            "hoverHighlight": False,
            "segments": [
                1,
            ],
            "segmentDefaultColor": self.color,
            "visible": self.is_visible,
        }


@dataclass
class ImageVolumeJSONGenerator(RenderingJSONGenerator):
    """Generates JSON Neuroglancer config for volume rendering."""

    color: str
    rendering_depth: int # Ideally, this should be a power of 2
    contrast_limits: tuple[float, float] = (-64, 64)
    threedee_contrast_limits: tuple[float, float] = (-64, 64)
    is_visible: bool = True

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def _get_shader(self) -> dict[str, Any]:
        # TODO (skm) - verify expected color usage
        shader_builder = ImageVolumeShaderBuilder(
            contrast_limits=self.contrast_limits,
            threedee_contrast_limits=self.threedee_contrast_limits,
            default_color=self.color,
        )
        return shader_builder.build_shader()

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": create_source(f"zarr://{self.source}", self.scale, self.scale),
            "tab": "rendering",
            "blend": "additive",
            "volumeRendering": "on",
            "volumeRenderingDepthSamples": self.rendering_depth,
            "visible": self.is_visible,
            **self._get_shader(),
        }
