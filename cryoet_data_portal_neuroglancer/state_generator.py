from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from scipy.spatial.transform import Rotation

from cryoet_data_portal_neuroglancer.models.json_generator import (
    AnnotationJSONGenerator,
    ImageJSONGenerator,
    ImageVolumeJSONGenerator,
    MeshJSONGenerator,
    OrientedPointAnnotationJSONGenerator,
    SegmentationJSONGenerator,
)
from cryoet_data_portal_neuroglancer.utils import get_scale


def _setup_creation(
    source: str,
    name: str | None = None,
    url: str | None = None,
    zarr_path: str | None = None,
    scale: tuple[float, float, float] | list[float] | float = 1.0,
) -> tuple[str, str, str, str, tuple[float, float, float]]:
    name = Path(source).stem if name is None else name
    url = url if url is not None else ""
    zarr_path = zarr_path if zarr_path is not None else source
    scale = get_scale(scale)
    source = urljoin(url, source.strip("/")) if url else source
    return source, name, url, zarr_path, scale


def _validate_color(color: str | None):
    if color and (len(color) != 7 or color[0] != "#"):
        raise ValueError("Color must be a hex string e.g. #FF0000")


def generate_point_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str = "#FFFFFF",
    point_size_multiplier: float = 1.0,
    scale: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    is_instance_segmentation: bool = False,
) -> dict[str, Any]:
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    _validate_color(color)
    return AnnotationJSONGenerator(
        source=source,
        name=name,
        color=color,
        point_size_multiplier=point_size_multiplier,
        scale=scale,
        is_visible=is_visible,
        is_instance_segmentation=is_instance_segmentation,
    ).to_json()


def generate_oriented_point_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str = "#FFFFFF",
    point_size_multiplier: float = 1.0,
    line_width: float = 1.0,
    scale: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    is_instance_segmentation: bool = False,
) -> dict[str, Any]:
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    _validate_color(color)
    return OrientedPointAnnotationJSONGenerator(
        source=source,
        name=name,
        color=color,
        point_size_multiplier=point_size_multiplier,
        line_width=line_width,
        scale=scale,
        is_visible=is_visible,
        is_instance_segmentation=is_instance_segmentation,
    ).to_json()


def generate_segmentation_mask_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str | None = "#FFFFFF",
    scale: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    display_bounding_box: bool = False,
    display_mesh: bool = True,
    highlight_on_hover: bool = False,
    mesh_render_scale: float = 2.0,
    visible_segments: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    _validate_color(color)
    return SegmentationJSONGenerator(
        source=source,
        name=name,
        color=color,
        scale=scale,
        is_visible=is_visible,
        display_bounding_box=display_bounding_box,
        display_mesh=display_mesh,
        highlight_on_hover=highlight_on_hover,
        mesh_render_scale=mesh_render_scale,
        visible_segments=visible_segments,
    ).to_json()


def generate_mesh_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str = "#FFFFFF",
    scale: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    display_bounding_box: bool = False,
    display_mesh: bool = True,
    highlight_on_hover: bool = False,
    mesh_render_scale: float = 2.0,
    visible_segments: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    _validate_color(color)
    return MeshJSONGenerator(
        source=source,
        name=name,
        color=color,
        scale=scale,
        is_visible=is_visible,
        display_bounding_box=display_bounding_box,
        display_mesh=display_mesh,
        highlight_on_hover=highlight_on_hover,
        mesh_render_scale=mesh_render_scale,
        visible_segments=visible_segments,
    ).to_json()


def generate_oriented_point_mesh_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str = "#FFFFFF",
    scale: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    display_bounding_box: bool = False,
    display_mesh: bool = True,
    highlight_on_hover: bool = False,
    mesh_render_scale: float = 4.0,
    visible_segments: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    return generate_mesh_layer(
        source=source,
        name=name,
        url=url,
        color=color,
        scale=scale,
        is_visible=is_visible,
        display_bounding_box=display_bounding_box,
        display_mesh=display_mesh,
        highlight_on_hover=highlight_on_hover,
        mesh_render_scale=mesh_render_scale,
        visible_segments=visible_segments,
    )


def generate_image_layer(
    source: str,
    scale: float | tuple[float, float, float],
    size: dict[str, float],
    name: str | None = None,
    url: str | None = None,
    start: dict[str, float] | None = None,
    mean: float | None = None,
    rms: float | None = None,
    is_visible: bool = True,
    has_volume_rendering_shader: bool = False,
) -> dict[str, Any]:
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    return ImageJSONGenerator(
        source=source,
        name=name,
        scale=scale,
        size=size,
        start=start,
        mean=mean,
        rms=rms,
        is_visible=is_visible,
        has_volume_rendering_shader=has_volume_rendering_shader,
    ).to_json()


def generate_image_volume_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str = "#FFFFFF",
    scale: tuple[float, float, float] | list[float] | float = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    rendering_depth: int = 1024,
) -> dict[str, Any]:
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    _validate_color(color)
    return ImageVolumeJSONGenerator(
        source=source,
        name=name,
        color=color,
        scale=scale,
        is_visible=is_visible,
        rendering_depth=rendering_depth,
    ).to_json()


def combine_json_layers(
    layers: list[dict[str, Any]],
    scale: tuple[float, float, float] | list[float] | float,
    units: str = "m",
    projection_quaternion: list[float] | None = None,
) -> dict[str, Any]:
    image_layers = [layer for layer in layers if layer["type"] == "image"]
    scale = get_scale(scale)
    if not projection_quaternion:
        projection_quaternion = Rotation.from_euler(seq="xyz", angles=(45, 0, 0), degrees=True).as_quat()
    combined_json = {
        "dimensions": {dim: [res, units] for dim, res in zip("xyz", scale, strict=False)},
        "crossSectionScale": 1.8,
        "projectionOrientation": list(projection_quaternion),
        "layers": layers,
        "selectedLayer": {
            "visible": True,
            "layer": layers[0]["name"],
        },
        "crossSectionBackgroundColor": "#000000",
        "layout": "4panel",
    }
    if len(image_layers) > 0 and "_position" in image_layers[0]:
        combined_json["position"] = image_layers[0]["_position"]
        combined_json["crossSectionScale"] = image_layers[0]["_crossSectionScale"]
        combined_json["projectionScale"] = image_layers[0]["_projectionScale"]

    return combined_json
