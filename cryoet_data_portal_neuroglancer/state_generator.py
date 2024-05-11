from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from utils import get_resolution


def setup_creation(
    source: str,
    name: Optional[str] = None,
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


def process_color(color: Optional[str]) -> tuple[str, str]:
    if color is None:
        return "vec4(255,255,255,1)", ""

    color_parts = color.split(" ")
    if len(color_parts) == 1:
        raise ValueError("Color must be a hex string followed by a name e.g. #FF0000 red")
    return color_parts[0], " ".join(color_parts[1:])


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
class AnnotationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: tuple[str, str]
    point_size_multiplier: float = 1.0
    oriented: bool = False

    def __post_init__(self):
        self._type = RenderingTypes.ANNOTATION

    def generate_json(self) -> dict:
        color_part = f" ({self.color[1]})" if self.color[1] else ""
        color_set = f"setColor({self.color[0]});\n"

        return {
            "type": self.layer_type,
            "name": f"{self.name}{color_part}",
            "source": f"precomputed://{self.source}",
            "tab": "rendering",
            "shader": f"#uicontrol float pointScale slider(min=0.01, max=2.0, default={self.point_size_multiplier}, step=0.01)\n"
            + "void main() {\n  "
            + color_set
            + "  setPointMarkerSize(pointScale * prop_diameter());\n"
            + "  setPointMarkerBorderWidth(0.1);\n"
            + "}",
        }



