from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .utils import get_resolution


def setup_creation(
        source: str,
        name: str = None,
        url: str = None,
        zarr_path: str = None,
        resolution: float | tuple[float, float, float] = None,
) -> tuple[str, str, str, str, tuple[float, float, float]]:
    name = Path(source).stem if name is None else name
    url = url if url is not None else ""
    zarr_path = zarr_path if zarr_path is not None else source
    resolution = get_resolution(resolution)
    sep = "/" if url else ""
    source = f"{url}{sep}{source}"
    return source, name, url, zarr_path, resolution


def make_transform(input_dict: dict, dim: str, resolution: float):
    input_dict[dim] = [resolution * 10e-10, "m"]


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
            raise ValueError(f"Unknown rendering type {self._type}")  # type: ignore

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
    contrast_limits: tuple[float, float] = (-64, 64)
    mean: float = None
    rms: float = None

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def create_shader_and_controls(self) -> dict[str, Any]:
        if self.mean is None or self.rms is None:
            distance = self.contrast_limits[1] - self.contrast_limits[0]
            window_start = self.contrast_limits[0] - (distance / 10)
            window_end = self.contrast_limits[1] + (distance / 10)
            shader = f"#uicontrol invlerp contrast(range=[{self.contrast_limits[0]}, {self.contrast_limits[1]}], " \
                     f"window=[{window_start}, {window_end}])\nvoid main() {{\n  emitGrayscale(contrast());\n}}"
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

    def generate_json(self) -> dict:
        transform: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1]):
            make_transform(transform, dim, resolution)

        original: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1]):
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
        return {**config, **self.create_shader_and_controls()}


@dataclass
class AnnotationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: str
    point_size_multiplier: float = 1.0

    def __post_init__(self):
        self._type = RenderingTypes.ANNOTATION

    def generate_json(self) -> dict:
        color_set = f"setColor({self.color});\n"

        return {
            "type": self.layer_type,
            "name": f"{self.name}",
            "source": f"precomputed://{self.source}",
            "tab": "rendering",
            "shader": f"#uicontrol float pointScale slider(min=0.01, max=2.0, default={self.point_size_multiplier}, step=0.01)\n"
                      + "void main() {\n  "
                      + color_set
                      + "  setPointMarkerSize(pointScale * prop_diameter());\n"
                      + "  setPointMarkerBorderWidth(0.1);\n"
                      + "}",
        }


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
