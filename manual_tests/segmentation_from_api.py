import json
import os
from pathlib import Path

from cryoet_data_portal import Annotation, Client, Tomogram
from scipy.spatial.transform import Rotation

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    PercentileContrastLimitCalculator,
    compute_contrast_limits,
)
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import encode_segmentation
from cryoet_data_portal_neuroglancer.state_generator import (
    combine_json_layers,
    generate_image_layer,
    generate_segmentation_mask_layer,
)

# logging.basicConfig(level=logging.DEBUG, force=True)

tomogram_path = "jsa2010-01-02-30.zarr"
zarr_path = "membrain_seg_prediction-1.0_segmentationmask.zarr"
output_path = "jsa2010-01-02-30-membrane-1.0_segmentationmask_encoded"
resolution = 1.564  # Non 1.0 is a harder test
# You could serve the tomogram locally instead, but the mesh has to be in
# a remote source like s3 or gs
SOURCE_MAP = (
    "s3://UPLOADED_TOMOGRAM_LOCATION",
    "s3://UPLOADED_ANNOTATION_LOCATION",
    (resolution * 1e-10, resolution * 1e-10, resolution * 1e-10),
)


def grab_tomogram():
    if not os.path.exists(tomogram_path):
        client = Client()
        tomogram = Tomogram.get_by_id(client, 13168)
        tomogram.download_omezarr()


def grab_annotation():
    if not os.path.exists(zarr_path):
        client = Client()
        annotation = Annotation.get_by_id(client, 28573)
        annotation.download(format="zarr")


def run_contrast_limit_calculations_from_api(input_data_path):
    data = load_omezarr_data(input_data_path, resolution_level=-1, persist=False)
    data_shape = data.shape
    data_size_dict = {"z": data_shape[0], "y": data_shape[1], "x": data_shape[2]}

    limits_dict = {}
    # Ensure the main public API is working
    gmm_limits = compute_contrast_limits(
        data,
        method="gmm",
    )
    limits_dict["gmm"] = gmm_limits
    limits_dict["2d"] = PercentileContrastLimitCalculator(data).compute_contrast_limit()
    limits_dict["size"] = data_size_dict
    return limits_dict


def make_precomputed_segmentation():
    encode_segmentation(
        zarr_path,
        output_path,
        (resolution, resolution, resolution),
        include_mesh=True,
        convert_non_zero_to=1,
        delete_existing=False,
        fast_bounding_box=True,
        labels_dict={1: "membrane"},
    )


def create_state(contrast_limit_dict, output_folder):
    source, seg_source, scale = SOURCE_MAP
    # One state layer for each contrast limit method
    layers_list = []
    ignored_keys = ["size", "closest_method", "distance_to_human", "2d"]

    # set the most promising contrast limits as visible, rest not
    for key, limits in contrast_limit_dict.items():
        if key in ignored_keys:
            continue
        is_visible = key == "gmm"
        layer_info = generate_image_layer(
            source=source,
            scale=scale,
            size=contrast_limit_dict["size"],
            name=f"{key}",
            twodee_contrast_limits=limits,
            threedee_contrast_limits=limits,
            has_volume_rendering_shader=True,
            volume_rendering_is_visible=True,
            is_visible=is_visible,
        )
        layer_info["_projectionScale"] = 2000
        layers_list.append(layer_info)
    seg_info = generate_segmentation_mask_layer(
        source=seg_source,
        scale=scale,
        name="membrane",
        color="#AA4500",
    )
    layers_list.append(seg_info)
    json_state = combine_json_layers(
        layers_list,
        scale,
        projection_quaternion=Rotation.from_euler(seq="xyz", angles=(0, 0, 0), degrees=True).as_quat(),
        show_axis_lines=False,
    )
    with open(output_folder / "state.json", "w") as f:
        json.dump(json_state, f, indent=4)
    return json_state


def main():
    grab_tomogram()
    grab_annotation()
    make_precomputed_segmentation()
    create_state(run_contrast_limit_calculations_from_api(tomogram_path), Path(os.getcwd()))


if __name__ == "__main__":
    main()
