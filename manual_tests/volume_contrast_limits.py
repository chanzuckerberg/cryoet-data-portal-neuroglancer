import argparse
import json
from pathlib import Path

from cryoet_data_portal import Client, Tomogram
from neuroglancer import viewer_state
from neuroglancer.url_state import to_url
from scipy.spatial.transform import Rotation

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    compute_contrast_limits,
)
from cryoet_data_portal_neuroglancer.state_generator import combine_json_layers, generate_image_layer

# Set up logging - level is info
# logging.basicConfig(level=logging.INFO, force=True)

OUTPUT_FOLDER = Path.cwd() / "volume_contrast_limits"

id_to_path_map = {
    773: "773/Position_513.zarr",
}

id_to_source_map = {
    773: (
        "https://files.cryoetdataportal.cziscience.com/10004/Position_513/Tomograms/VoxelSpacing7.560/CanonicalTomogram/Position_513.zarr",
        (756e-10, 756e-10, 756e-10),
    ),
}


def grab_tomogram(id_: int, zarr_path: Path):
    client = Client()
    if not zarr_path.exists():
        zarr_path.mkdir(parents=True, exist_ok=True)
        tomogram = Tomogram.get_by_id(client, id_)
        tomogram.download_omezarr(str(zarr_path.parent.resolve()))


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
    cdf_limits = compute_contrast_limits(
        data,
        method="cdf",
    )
    limits_dict["gmm"] = gmm_limits
    limits_dict["cdf"] = cdf_limits
    limits_dict["2d"] = gmm_limits
    limits_dict["size"] = data_size_dict
    return limits_dict


def create_state(id_, contrast_limit_dict, output_folder):
    source, scale = id_to_source_map[id_]
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
            name=f"{id_}_{key}",
            twodee_contrast_limits=limits,
            threedee_contrast_limits=limits,
            has_volume_rendering_shader=True,
            volume_rendering_is_visible=True,
            is_visible=is_visible,
        )
        layer_info["_projectionScale"] = 2000
        layers_list.append(layer_info)
    json_state = combine_json_layers(
        layers_list,
        scale,
        projection_quaternion=Rotation.from_euler(seq="xyz", angles=(0, 0, 0), degrees=True).as_quat(),
        show_axis_lines=False,
    )
    with open(output_folder / f"{id_}_state.json", "w") as f:
        json.dump(json_state, f, indent=4)
    return json_state


def main(output_folder):
    url_list = []
    for id_, path in id_to_path_map.items():
        path = Path(output_folder) / path
        grab_tomogram(id_, path)
        limits = run_contrast_limit_calculations_from_api(path)
        state = create_state(id_, limits, Path(output_folder) / "results")
        viewer_state_obj = viewer_state.ViewerState(state)
        url_from_json = to_url(
            viewer_state_obj,
            prefix="https://neuroglancer-demo.appspot.com/",
        )
        url_list.append(url_from_json)
    print(f"Wrote {len(url_list)} urls to {Path(output_folder) / 'urls.txt'}")
    with open(Path(output_folder) / "urls.txt", "w") as f:
        f.write("\n\n\n".join(url_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    args, _ = parser.parse_known_args()
    main(args.output_folder)
