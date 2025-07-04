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
class SegmentPropertyJSONGenerator:
    """Generates a JSON file for segmentation properties.

    Supports a subset of the properties that can be set in Neuroglancer.
    See https://github.com/google/neuroglancer/blob/3efc90465e702453916d2b03d472c16378848132/src/datasource/precomputed/segment_properties.md
    """

    ids: list[int]
    labels: list[str]

    def generate_json(self) -> dict:
        return {
            "inline": {
                "ids": [str(val) for val in self.ids],
                "properties": [
                    {
                        "values": self.labels,
                        "type": "label",
                        "id": "label",
                    },
                ],
            },
            "@type": "neuroglancer_segment_properties",
        }


@dataclass
class RenderingJSONGenerator:
    """Generates a JSON file for Neuroglancer to read."""

    source: str
    name: str
    scale: tuple[float, float, float]
    output_scale: tuple[float, float, float] | None

    def __post_init__(self):
        if self.output_scale is None:
            self.output_scale = self.scale

    def create_source(self, source_prefix: str) -> dict[str, Any]:
        """Creates the source dictionary for Neuroglancer."""
        return create_source(
            f"{source_prefix}://{self.source}",
            self.scale,
            self.output_scale,
        )

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
    contrast_limits: tuple[float, float] | None = None
    threedee_contrast_limits: tuple[float, float] | None = None
    start: dict[str, float] | None = None
    mean: float | None = None
    rms: float | None = None
    is_visible: bool = True
    has_volume_rendering_shader: bool = False
    volume_rendering_depth_samples: int = 128  # Ideally, this should be a power of 2
    volume_rendering_is_visible: bool = False
    volume_rendering_gain: float = 0.0
    can_hide_high_values_in_neuroglancer: bool = False
    blend: str = "additive"
    opacity: float = 1.0
    is_code_editor_visible: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._type = RenderingTypes.IMAGE

    def _compute_contrast_limits(self) -> tuple[float, float]:
        if self.mean is None or self.rms is None:
            # return self.contrast_limits
            return (0.0, 1.0)
        width = 3 * self.rms
        return (self.mean - width, self.mean + width)

    def _create_shader_and_controls(self) -> dict[str, Any]:
        if self.contrast_limits is None:
            self.contrast_limits = self._compute_contrast_limits()
        if self.threedee_contrast_limits is None:
            self.threedee_contrast_limits = self.contrast_limits
        if self.has_volume_rendering_shader:
            shader_builder = ImageWithVolumeRenderingShaderBuilder(
                contrast_limits=self.contrast_limits,
                threedee_contrast_limits=self.threedee_contrast_limits,
                can_hide_high_values_in_neuroglancer=self.can_hide_high_values_in_neuroglancer,
            )
        else:
            shader_builder = ImageShaderBuilder(
                contrast_limits=self.contrast_limits,
                # can_hide_high_values_in_neuroglancer=self.can_hide_high_values_in_neuroglancer,
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
            "source": self.create_source("zarr"),
            "opacity": self.opacity,
            "blend": self.blend,
            "tab": "rendering",
            "visible": self.is_visible,
            "volumeRendering": "on" if self.volume_rendering_is_visible else "off",
            "volumeRenderingGain": self.volume_rendering_gain,
            "codeVisible": self.is_code_editor_visible,
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
    is_code_editor_visible: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._type = RenderingTypes.ANNOTATION

    def _get_shader(self):
        shader_builder = NonOrientedPointShaderBuilder(
            point_size_multiplier=self.point_size_multiplier,
            is_instance_segmentation=self.is_instance_segmentation,
        )
        return shader_builder.build()

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": self.create_source("precomputed"),
            "tab": "rendering",
            "annotationColor": self.color,
            "visible": self.is_visible,
            "codeVisible": self.is_code_editor_visible,
            **self._get_shader(),
        }


@dataclass
class OrientedPointAnnotationJSONGenerator(AnnotationJSONGenerator):
    """Generates JSON Neuroglancer config for oriented point annotation."""

    line_width: float = 1.0

    def _get_shader(self):
        shader_builder = OrientedPointShaderBuilder(
            point_size_multiplier=self.point_size_multiplier,
            is_instance_segmentation=self.is_instance_segmentation,
            line_width=self.line_width,
        )
        return shader_builder.build()


@dataclass
class SegmentationJSONGenerator(RenderingJSONGenerator):
    """Generates JSON Neuroglancer config for segmentation mask."""

    color: str | None = None
    is_visible: bool = True
    display_mesh: bool = True
    display_bounding_box: bool = False
    highlight_on_hover: bool = False
    mesh_render_scale: float = 1.0
    visible_segments: tuple[int, ...] = (1,)
    enable_pick: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._type = RenderingTypes.SEGMENTATION

    def generate_json(self) -> dict:
        state = {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": {
                **self.create_source("precomputed"),
                "subsources": {
                    "default": True,
                    "mesh": self.display_mesh,
                },
                "enableDefaultSubsources": self.display_bounding_box,
            },
            "tab": "rendering",
            "selectedAlpha": 1,
            "hoverHighlight": self.highlight_on_hover,
            "segments": sorted((self.visible_segments)),
            "visible": self.is_visible,
            "meshRenderScale": self.mesh_render_scale,
            "pick": self.enable_pick,
        }
        # self.color === None means that the color will be random
        # This is useful for multiple segmentations
        if self.color is not None:
            state["segmentDefaultColor"] = self.color
        return state


# Alias for SegmentationJSONGenerator - for future compatibility
MeshJSONGenerator = SegmentationJSONGenerator


@dataclass
class ImageVolumeJSONGenerator(RenderingJSONGenerator):
    """Generates JSON Neuroglancer config for volume rendering."""

    color: str
    rendering_depth: int
    blend: str = "additive"
    opacity: float = 1.0

    def __post_init__(self):
        super().__post_init__()
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
            "source": self.create_source("zarr"),
            "tab": "rendering",
            "blend": self.blend,
            "opacity": self.opacity,
            "volumeRendering": "on",
            "volumeRenderingDepthSamples": self.rendering_depth,
            "visible": self.is_visible,
            **self._get_shader(),
        }
