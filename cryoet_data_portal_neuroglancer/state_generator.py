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
    """Starting point for generating JSON for a layer.

    By taking in some possibly optional parameters for the data setup
    , we combine these in a way that is useful for the rest of the code
    around state generation.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    zarr_path: str | None, optional
        The path to the zarr data. If None, the source will be used.
        This is useful where the source is a URL, not a file path.
    scale: float | tuple[float, float, float], optional
        The scale/resolution of the data in metres. Ordering is XYZ.
        If a single float is provided, it will be used for all three axes.
        Default is (1.0, 1.0, 1.0).

    Returns
    -------
    tuple[str, str, str, str, tuple[float, float, float]]
        A tuple containing the source, name, url, zarr_path, and scale.
    """
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
    """Generate neuroglancer JSON state for a single point annotation layer.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    color: str, optional
        The color of the points in hex format. Default is white (#FFFFFF).
    point_size_multiplier: float, optional
        The size of the points in the annotation layer. Default is 1.0.
    scale: float | tuple[float, float, float], optional
        The scale/resolution of the data in metres. Ordering is XYZ.
        If a single float is provided, it will be used for all three axes.
        Default is (1.0, 1.0, 1.0).
    is_visible: bool, optional
        Whether the layer should be visible in the viewer on startup. Default is True.
    is_instance_segmentation: bool, optional
        An instance segmentation layer is a layer where each point is assigned to a specific instance.
        This is useful for layers where you want to assign different colors to different instances.
        Default is False, which means each point is assigned the same color.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the JSON state for the layer.

    Raises
    ------
    ValueError
        If the color is not a valid hex string.
    """
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
    is_code_editor_visible: bool = False,
) -> dict[str, Any]:
    """Generate neuroglancer JSON state for a single oriented point annotation layer.

    Oriented point annotation layers consist of points shown as three direction vectors.
    The direction vectors are line annotations.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    color: str, optional
        The color of the points in hex format. Default is white (#FFFFFF).
    point_size_multiplier: float, optional
        The size of the points in the annotation layer. Default is 1.0.
    line_width: float, optional
        The width of the lines in the annotation layer. Default is 1.0.
    scale: float | tuple[float, float, float], optional
        The scale/resolution of the data in metres. Ordering is XYZ.
        If a single float is provided, it will be used for all three axes.
        Default is (1.0, 1.0, 1.0).
    is_visible: bool, optional
        Whether the layer should be visible in the viewer on startup. Default is True.
    is_instance_segmentation: bool, optional
        An instance segmentation layer is a layer where each point is assigned to a specific instance.
        This is useful for layers where you want to assign different colors to different instances.
        Default is False, which means each point is assigned the same color.
    is_code_editor_visible: bool, optional
        Whether the code editor should be visible in the layer settings. Default is False.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the JSON state for the layer.

    Raises
    ------
    ValueError
        If the color is not a valid hex string.
    """
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
        is_code_editor_visible=is_code_editor_visible,
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
    enable_pick: bool = False,
) -> dict[str, Any]:
    """Generate neuroglancer JSON state for a single segmentation layer.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    color: str, optional
        The color of the points in hex format. Default is white (#FFFFFF).
    scale: float | tuple[float, float, float], optional
        The scale/resolution of the data in metres. Ordering is XYZ.
        If a single float is provided, it will be used for all three axes.
        Default is (1.0, 1.0, 1.0).
    is_visible: bool, optional
        Whether the layer should be visible in the viewer on startup. Default is True.
    display_bounding_box: bool, optional
        Whether to display the bounding box of the layer. Default is False.
    display_mesh: bool, optional
        Whether to display the mesh version of the mask. Default is True.
    highlight_on_hover: bool, optional
        Whether to highlight the mesh on hover. Default is False.
    mesh_render_scale: float, optional
        The scale of the mesh rendering. Default is 2.0.
        This is the scale of the mesh rendering in the viewer.
        A lower value will force a higher resolution mesh to be loaded
        in general.
    visible_segments: tuple[int, ...], optional
        The labels segments that should be visible in the viewer. Default is (1,).
        Many of the segmentation masks have only one segment, which is
        labeled as 1. So (1,) - the default, shows just that segment.
    enable_pick: bool, optional
        Technically, this is the "pick" option in neuroglancer.
        Having it off also has the effect of making it so that
        double clicking the segmentation mask does not turn off the
        segmentation mask in the viewer.
        Default is False.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the JSON state for the layer.

    Raises
    ------
    ValueError
        If the color is not a valid hex string.
    """
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
        enable_pick=enable_pick,
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
    """Generate neuroglancer JSON state for a single mesh.

    Techically, this is a "segmentation" layer in neuroglancer.
    But all the layer shows is a mesh, so we call it a mesh layer.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    color: str, optional
        The color of the points in hex format. Default is white (#FFFFFF).
    scale: float | tuple[float, float, float], optional
        The scale/resolution of the data in metres. Ordering is XYZ.
        If a single float is provided, it will be used for all three axes.
        Default is (1.0, 1.0, 1.0).
    is_visible: bool, optional
        Whether the layer should be visible in the viewer on startup. Default is True.
    display_bounding_box: bool, optional
        Whether to display the bounding box of the layer. Default is False.
    display_mesh: bool, optional
        Whether to display the mesh version of the mask. Default is True.
    highlight_on_hover: bool, optional
        Whether to highlight the mesh on hover. Default is False.
    mesh_render_scale: float, optional
        The scale of the mesh rendering. Default is 2.0.
        This is the scale of the mesh rendering in the viewer.
        A lower value will force a higher resolution mesh to be loaded
        in general.
    visible_segments: tuple[int, ...], optional
        The labels segments that should be visible in the viewer. Default is (1,).
        Many of the segmentation masks have only one segment, which is
        labeled as 1. So (1,) - the default, shows just that segment.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the JSON state for the point annotation layer.

    Raises
    ------
    ValueError
        If the color is not a valid hex string.
    """
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
    """Generate neuroglancer JSON state for a multiple instanced oriented meshes.

    Techically, this is a "segmentation" layer in neuroglancer.
    But all the layer shows is a mesh, so we call it a mesh layer.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    color: str, optional
        The color of the points in hex format. Default is white (#FFFFFF).
    scale: float | tuple[float, float, float], optional
        The scale/resolution of the data in metres. Ordering is XYZ.
        If a single float is provided, it will be used for all three axes.
        Default is (1.0, 1.0, 1.0).
    is_visible: bool, optional
        Whether the layer should be visible in the viewer on startup. Default is True.
    display_bounding_box: bool, optional
        Whether to display the bounding box of the layer. Default is False.
    display_mesh: bool, optional
        Whether to display the mesh version of the mask. Default is True.
    highlight_on_hover: bool, optional
        Whether to highlight the mesh on hover. Default is False.
    mesh_render_scale: float, optional
        The scale of the mesh rendering. Default is 2.0.
        This is the scale of the mesh rendering in the viewer.
        A lower value will force a higher resolution mesh to be loaded
        in general.
    visible_segments: tuple[int, ...], optional
        The labels segments that should be visible in the viewer. Default is (1,).
        Many of the segmentation masks have only one segment, which is
        labeled as 1. So (1,) - the default, shows just that segment.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the JSON state for the layer.

    Raises
    ------
    ValueError
        If the color is not a valid hex string.
    """
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
    has_volume_rendering_shader: bool | None = None,
    twodee_contrast_limits: tuple[float, float] | None = None,
    threedee_contrast_limits: tuple[float, float] | None = None,
    volume_rendering_is_visible: bool = False,
    volume_rendering_gain: float = -7.8,
    can_hide_high_values_in_neuroglancer: bool | None = None,
    blend: str = "additive",
    opacity: float = 1.0,
    is_code_editor_visible: bool = False,
) -> dict[str, Any]:
    """Generate neuroglancer JSON state for a tomogram (image) layer.

    Parameters
    ----------
    source: str
        The source of the data. This can be a file path or a direct URL.
    scale: float | tuple[float, float, float]
        The scale/resolution of the data in metres. Ordering is XYZ.
    size: dict[str, float]
        The size of the data in each dimension in in pixels.
        (e.g. {"x": 512, "y": 400, "z": 256}).
        The keys are "x", "y", and "z".
        This is used to set the center position of the data in the viewer
        and the zoom level in 2D and 3D views.
    name: str | None, optional
        The name of the layer. If None, the name will be derived from the source.
    url: str | None, optional
        The base URL for the data. If None, the source must be a direct URL.
    start: dict[str, float] | None, optional
        The starting position of the data, usually not needed and can be left as None.
        Default is None, which is essentially the same as {"x": 0, "y": 0, "z": 0}.
        The keys are "x", "y", and "z".
    mean: float | None, optional
        The mean value of the data. Default is None.
    rms: float | None, optional
        The root mean square value of the data. Default is None.
        If contrast limits are not provided, at least one of the mean or rms
        should be provided.
    is_visible: bool, optional
        Whether the layer should be visible in the viewer on startup. Default is True.
    has_volume_rendering_shader: bool | None, optional
        Whether to create the layer with a volume rendering shader.
        Default is None, which means it will be set to True if
        volume_rendering_is_visible is True or threedee_contrast_limits
        is not None.
    twodee_contrast_limits: tuple[float, float] | None, optional
        The contrast limits for the 2D view. Default is None.
        If None, the mean and rms values will be used to set the contrast limits.
        If both mean and rms are None, the contrast limits will be set to
        (0, 1) by default.
    threedee_contrast_limits: tuple[float, float] | None, optional
        The contrast limits for the 3D view. Default is None.
        If None, these default to the same as the 2D contrast limits.
    volume_rendering_is_visible: bool, optional
        Whether to show the volume rendering in the viewer. Default is False.
    volume_rendering_gain: float, optional
        The gain for the volume rendering. Default is -7.8.
        This is an exponential gain value that is used to adjust the alpha
        of the volume rendering. A lower value will make the volume rendering
        darker / more transparent, while a higher value will make it brighter and
        more opaque.
    can_hide_high_values_in_neuroglancer: bool | None, optional
        Whether to allow the user to hide high values in the neuroglancer viewer.
        High values are defined as values above the contrast limits.
        Having this set to True spawns a control in the layer side panel
        that allows the user to hide high values for this layer.
        Default is None, which means it will be set to True if
        has_volume_rendering_shader is True.
    blend: str, optional
        The blending mode for the layer. Default is "additive".
        The only other option is "default"
    opacity: float, optional
        The opacity of the layer. Default is 1.0.
        This is the opacity of the layer in the viewer.
        A lower value will make the layer more transparent, while a higher
        value will make it more opaque.
    is_code_editor_visible: bool, optional
        Whether the code editor should be visible in the layer settings.
        Default is False.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the JSON state for the layer.
    """
    # If volume rendering is visible, set the flag to True for the relevant shader
    if has_volume_rendering_shader is None:
        has_volume_rendering_shader = volume_rendering_is_visible or threedee_contrast_limits is not None
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    if can_hide_high_values_in_neuroglancer is None:
        can_hide_high_values_in_neuroglancer = has_volume_rendering_shader
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
        contrast_limits=twodee_contrast_limits,
        threedee_contrast_limits=threedee_contrast_limits,
        volume_rendering_is_visible=volume_rendering_is_visible,
        volume_rendering_gain=volume_rendering_gain,
        can_hide_high_values_in_neuroglancer=can_hide_high_values_in_neuroglancer,
        blend=blend,
        opacity=opacity,
        is_code_editor_visible=is_code_editor_visible,
    ).to_json()


def generate_image_volume_layer(
    source: str,
    name: str | None = None,
    url: str | None = None,
    color: str = "#FFFFFF",
    scale: tuple[float, float, float] | list[float] | float = (1.0, 1.0, 1.0),
    is_visible: bool = True,
    rendering_depth: int = 1024,
    blend: str = "additive",
    opacity: float = 1.0,
) -> dict[str, Any]:
    """
    ! This function is deprecated, please use "generate_segmentation_mask_layer" instead !

    It was originally used to visualize segmentations via volume rendering.
    """
    source, name, url, _, scale = _setup_creation(source, name, url, scale=scale)
    _validate_color(color)
    return ImageVolumeJSONGenerator(
        source=source,
        name=name,
        color=color,
        scale=scale,
        is_visible=is_visible,
        rendering_depth=rendering_depth,
        blend=blend,
        opacity=opacity,
    ).to_json()


def combine_json_layers(
    layers: list[dict[str, Any]],
    scale: tuple[float, float, float] | list[float] | float,
    units: str = "m",
    projection_quaternion: list[float] | None = None,
    set_slices_visible_in_3d: bool | None = None,
    show_axis_lines: bool = True,
    enable_layer_color_legend: bool = True,
    use_old_neuroglancer_layout: bool = True,
    gpu_memory_limit_gb: float = 1.5,
) -> dict[str, Any]:
    """Take a list of JSON layers and combine them into a neuroglancer state.

    Parameters
    ----------
    layers: list[dict[str, Any]]
        The list of JSON state neuroglancer layers to combine.
    scale: float | tuple[float, float, float] | list[float]
        The scale/resolution for the global neuroglancer co-ordinate space in "units". Ordering is XYZ.
    units: str, optional
        The units for the scale. Default is "m".
        For example, "nm" for nanometres, "um" for micrometres, etc.
    projection_quaternion: list[float] | None, optional
        The quaternion for the projection orientation. Default is None.
        If None, the default is a rotation of 45 degrees around the x-axis.
        Easiest is to use the scipy.spatial.transform.Rotation class to generate
        the quaternion. For example:
        >>> from scipy.spatial.transform import Rotation
        >>> projection_quaternion = Rotation.from_euler(seq="xyz", angles=(45, 0, 0), degrees=True).as_quat()
    set_slices_visible_in_3d: bool | None, optional
        Whether to show the slices in 3D. Default is None.
        If None, it will be set to False if there are any image layers in the list with volume rendering.
    show_axis_lines: bool, optional
        Whether to show the axis lines in the 3D view. Default is True.
    enable_layer_color_legend: bool, optional
        Whether to enable the layer color legend in the 3D view. Default is True.
    use_old_neuroglancer_layout: bool, optional
        Whether to use the old neuroglancer panels layout. Default is True.
    gpu_memory_limit_gb: float, optional
        The GPU memory limit for the neuroglancer viewer in GB. Default is 1.5.
        Going above 2GB would not be recommended as it may cause the browser to crash.
        In between 1.5 and 2GB could be experimented with.
        The default neuroglancer uses is actually 1GB GPU memory, and 2GB system memory.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the combined JSON state for the layers.

    Raises
    ------
    ValueError
        If the scale is not a valid tuple or list of floats.
    """
    image_layers = [layer for layer in layers if layer["type"] == "image"]
    if set_slices_visible_in_3d is None:
        set_slices_visible_in_3d = not any(layer["volumeRendering"] == "on" for layer in image_layers)

    scale = get_scale(scale)
    if projection_quaternion is None:
        projection_quaternion = Rotation.from_euler(seq="xyz", angles=(45, 0, 0), degrees=True).as_quat()

    layout = "4panel" if use_old_neuroglancer_layout else "4panel-alt"
    combined_json = {
        "dimensions": {dim: [res, units] for dim, res in zip("xyz", scale, strict=False)},
        "crossSectionScale": 1.8,
        "projectionOrientation": list(projection_quaternion),
        "layers": layers,
        "selectedLayer": {
            "visible": True,
            "layer": layers[0]["name"],
            "side": "left",
        },
        "crossSectionBackgroundColor": "#000000",
        "hideCrossSectionBackground3D": True,
        "layout": layout,
        "showSlices": set_slices_visible_in_3d,
        "showAxisLines": show_axis_lines,
        "enableLayerColorWidget": enable_layer_color_legend,
        "layerListPanel": {
            "row": 1,
            "visible": True,
        },
        "helpPanel": {
            "side": "right",
            "row": 0,
        },
        "settingsPanel": {
            "side": "right",
            "row": 1,
        },
        "selection": {
            "row": 2,
            "visible": False,
        },
        "gpuMemoryLimit": int(gpu_memory_limit_gb * 1e9),
    }
    if len(image_layers) > 0 and "_position" in image_layers[0]:
        combined_json["position"] = image_layers[0]["_position"]
        combined_json["crossSectionScale"] = image_layers[0]["_crossSectionScale"]
        combined_json["projectionScale"] = image_layers[0]["_projectionScale"]

    return combined_json
