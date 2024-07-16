from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from cryoet_data_portal_neuroglancer.shaders.annotation import (
    NonOrientedPointShaderBuilder,
    OrientedPointShaderBuilder,
)
from cryoet_data_portal_neuroglancer.shaders.image import (
    ImageShaderBuilder,
    ImageWithVolumeRenderingShaderBuilder,
)


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
    start: dict[str, float] | None = None
    mean: float | None = None
    rms: float | None = None
    is_visible: bool = True
    has_volume_rendering_shader: bool = False
    volume_rendering_depth_samples: int = 256  # Ideally, this should be a power of 2

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def _compute_contrast_limits(self) -> tuple[float, float]:
        if self.mean is None or self.rms is None:
            return self.contrast_limits
        width = 3 * self.rms
        return (self.mean - width, self.mean + width)

    def _create_shader_and_controls(self) -> dict[str, Any]:
        contrast_limits = self._compute_contrast_limits()
        if self.has_volume_rendering_shader:
            # At the moment these are the same limits,
            # but in the future the calculation might change for 3D rendering
            threedee_contrast_limits = contrast_limits
            shader_builder = ImageWithVolumeRenderingShaderBuilder(
                contrast_limits=contrast_limits,
                threedee_contrast_limits=threedee_contrast_limits,
            )
        else:
            shader_builder = ImageShaderBuilder(
                contrast_limits=contrast_limits,
            )
        return shader_builder.build()

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
        if self.has_volume_rendering_shader:
            config["volumeRenderingDepthSamples"] = self.volume_rendering_depth_samples
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
        shader_builder = NonOrientedPointShaderBuilder(
            point_size_multiplier=self.point_size_multiplier,
            is_instance_segmentation=self.is_instance_segmentation,
            color=self.color,
        )
        return shader_builder.build()

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": create_source(f"precomputed://{self.source}", self.scale, self.scale),
            "tab": "rendering",
            "visible": self.is_visible,
            **self._get_shader(),
        }


@dataclass
class OrientedPointAnnotationGenerator(AnnotationJSONGenerator):
    """Generates JSON Neuroglancer config for oriented point annotation."""

    line_width: float = 1.0

    def _get_shader(self):
        shader_builder = OrientedPointShaderBuilder(
            point_size_multiplier=self.point_size_multiplier,
            is_instance_segmentation=self.is_instance_segmentation,
            color=self.color,
            line_width=self.line_width,
        )
        return shader_builder.build()


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
    rendering_depth: int

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def _get_shader(self) -> dict[str, Any]:
        shader = (
            f'#uicontrol vec3 color color(default="{self.color}")\n'
            f"#uicontrol invlerp toRaw(range=[0, 1], window=[-1, 2])\n"
            f"void main() {{\n"
            f"  emitRGBA(vec4(color * toRaw(getDataValue()), toRaw(getDataValue())));\n"
            f"}}"
        )
        return {
            "shader": shader,
            "shaderControls": {"color": self.color},
        }

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
