from pathlib import Path
from typing import Any, Optional

from cryoet_data_portal_neuroglancer.models.json_generator import (
    AnnotationJSONGenerator,
    ImageJSONGenerator,
    SegmentationJSONGenerator,
)
from cryoet_data_portal_neuroglancer.utils import get_resolution


def setup_creation(
    source: str,
    name: str = None,
    url: str = None,
    zarr_path: str = None,
    resolution: float | tuple[float, float, float] = 1.0,
) -> tuple[str, str, str, str, tuple[float, float, float]]:
    name = Path(source).stem if name is None else name
    url = url if url is not None else ""
    zarr_path = zarr_path if zarr_path is not None else source
    resolution = get_resolution(resolution)
    sep = "/" if url else ""
    source = f"{url}{sep}{source}"
    return source, name, url, zarr_path, resolution


def _parse_to_vec4_color(input_color: list[str]) -> str:
    """
    Parse the color from a list of strings to a webgl vec4 of rgba color
    """
    if input_color is None:
        output_color = [255, 255, 255]
    elif len(input_color) == 1:
        color = input_color[0]
        output_color = [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
    elif len(input_color) == 3:
        output_color = [int(x) for x in input_color]  # type: ignore
    else:
        raise ValueError(f"Color must be a list of 3 values, provided: {input_color}")
    output_color.append(255)
    return f"vec4({output_color[0]}, {output_color[1]}, {output_color[2]}, {output_color[3]})"


def _parse_to_hex_color(color: Optional[str]) -> tuple[str, str]:
    if color is None:
        return "#FFFFFF", ""
    color_parts = color.split(" ")
    if len(color_parts) == 1:
        raise ValueError("Color must be a hex string followed by a name e.g. #FF0000 red")
    return color_parts[0], " ".join(color_parts[1:])


def generate_point_layer(
    source: str,
    name: str = None,
    url: str = None,
    color: list[str] = None,
    point_size_multiplier: float = 1.0,
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    source, name, url, _, resolution = setup_creation(source, name, url, resolution=resolution)
    new_color = _parse_to_vec4_color(color)
    return AnnotationJSONGenerator(
        source=source,
        name=name,
        color=new_color,
        point_size_multiplier=point_size_multiplier,
        resolution=resolution,
    ).to_json()


def generate_segmentation_mask_layer(
    source: str,
    name: str = None,
    url: str = None,
    color: str = None,
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    source, name, url, _, resolution = setup_creation(source, name, url, resolution=resolution)
    color_tuple = _parse_to_hex_color(color)
    return SegmentationJSONGenerator(source=source, name=name, color=color_tuple, resolution=resolution).to_json()


def generate_image_layer(
    source: str,
    resolution: tuple[float, float, float],
    size: dict[str, float],
    name: str = None,
    url: str = None,
    start: dict[str, float] = None,
    mean: float = None,
    rms: float = None,
) -> dict[str, Any]:
    source, name, url, _, resolution = setup_creation(source, name, url, resolution=resolution)
    return ImageJSONGenerator(
        source=source,
        name=name,
        resolution=resolution,
        size=size,
        start=start,
        mean=mean,
        rms=rms,
    ).to_json()


def combine_json_layers(
    layers: list[dict[str, Any]],
    resolution: Optional[tuple[float, float, float] | list[float]] = None,
    units: str = "m",
) -> dict[str, Any]:
    image_layers = [layer for layer in layers if layer["type"] == "image"]
    resolution = get_resolution(resolution)
    dimensions = {dim: [res, units] for dim, res in zip("xyz", resolution, strict=False)}

    combined_json = {
        "dimensions": dimensions,
        "crossSectionScale": 1.8,
        "projectionOrientation": [0.173, -0.0126, -0.0015, 0.984],
        "layers": layers,
        "selectedLayer": {
            "visible": True,
            "layer": layers[0]["name"],
        },
        "crossSectionBackgroundColor": "#000000",
        "layout": "4panel",
    }
    if image_layers is not None:
        combined_json["position"] = image_layers[0]["_position"]
        combined_json["crossSectionScale"] = image_layers[0]["_crossSectionScale"]

    return combined_json
