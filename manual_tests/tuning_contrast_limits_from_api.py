import argparse
import json
from pathlib import Path

import numpy as np
from cryoet_data_portal import Client, Tomogram
from neuroglancer import viewer_state
from neuroglancer.url_state import to_url
from scipy.spatial.transform import Rotation
from screenshot_urls import run_screenshot_loop

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    CDFContrastLimitCalculator,
    ContrastLimitCalculator,
    GMMContrastLimitCalculator,
    combined_contrast_limit_plot,
    compute_contrast_limits,
)
from cryoet_data_portal_neuroglancer.state_generator import combine_json_layers, generate_image_layer

# Set up logging - level is info
# logging.basicConfig(level=logging.INFO, force=True)

OUTPUT_FOLDER = Path.cwd() / "contrast_limits_from_api"

id_to_path_map = {
    1000: "1000/16.zarr",
    706: "706/Position_161.zarr",
    800: "800/0105.zarr",
    10845: "10845/ay18112021_grid2_lamella3_position7.zarr",
    4279: "4279/dga2018-08-27-600.zarr",
}

id_to_human_contrast_limits = {
    1000: {
        "slice": [-0.2, 0.15],
        "volume": [-0.035, 0.009],
        "gain": -7.6,
    },
    706: {
        "slice": [-44499.8, 83143],
        "volume": [-20221.2, 18767.6],
        "gain": -7.7,
    },
    800: {
        "slice": [0.0000748111, 0.00189353],
        "volume": [0.000705811, 0.00152511],
        "gain": -8.6,
    },
    10845: {
        "slice": [-29.5498, 47.7521],
        "volume": [-11.0534, 20.0755],
        "gain": -7.9,
    },
    4279: {
        "slice": [0.580756, 31.9362],
        "volume": [15.5439, 22.0607],
        "gain": -7.5,
    },
}

id_to_source_map = {
    1000: (
        "https://files.cryoetdataportal.cziscience.com/10008/16/Tomograms/VoxelSpacing14.080/CanonicalTomogram/16.zarr",
        (1.408e-9, 1.408e-9, 1.408e-9),
    ),
    706: (
        "https://files.cryoetdataportal.cziscience.com/10004/Position_161/Tomograms/VoxelSpacing7.560/CanonicalTomogram/Position_161.zarr",
        (7.56e-10, 7.56e-10, 7.56e-10),
    ),
    800: (
        "https://files.cryoetdataportal.cziscience.com/10005/0105/Tomograms/VoxelSpacing5.224/CanonicalTomogram/0105.zarr",
        (5.224e-10, 5.224e-10, 5.224e-10),
    ),
    10845: (
        "https://files.cryoetdataportal.cziscience.com/10007/ay18112021_grid2_lamella3_position7/Tomograms/VoxelSpacing7.840/CanonicalTomogram/ay18112021_grid2_lamella3_position7.zarr",
        (7.84e-10, 7.84e-10, 7.84e-10),
    ),
    4279: (
        "https://files.cryoetdataportal.cziscience.com/10059/dga2018-08-27-600/Tomograms/VoxelSpacing16.800/CanonicalTomogram/dga2018-08-27-600.zarr",
        (1.68e-9, 1.68e-9, 1.68e-9),
    ),
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def grab_tomogram(id_: int, zarr_path: Path):
    client = Client()
    if not zarr_path.exists():
        zarr_path.mkdir(parents=True, exist_ok=True)
        tomogram = Tomogram.get_by_id(client, id_)
        tomogram.download_omezarr(str(zarr_path.parent.resolve()))


def run_contrast_limit_calculations_from_api(
    id_,
    input_data_path,
    output_path,
    z_radius=5,
    ds_factor=0.3,
):
    output_path.mkdir(parents=True, exist_ok=True)
    human_contrast = id_to_human_contrast_limits[id_]
    volume_limit = human_contrast["volume"]
    data = load_omezarr_data(input_data_path, resolution_level=-1, persist=False)
    data_shape = data.shape
    data_size_dict = {"z": data_shape[0], "y": data_shape[1], "x": data_shape[2]}

    limits_dict = {}
    # Ensure the main public API is working
    gmm_limits = compute_contrast_limits(
        data,
        method="gmm",
        z_radius=z_radius,
        downsampling_ratio=ds_factor,
    )
    cdf_limits = compute_contrast_limits(
        data,
        method="cdf",
        z_radius=z_radius,
        downsampling_ratio=ds_factor,
    )
    limits_dict["gmm"] = gmm_limits
    limits_dict["cdf"] = cdf_limits

    # From here on is saving the results and extra info to make a URL state
    with open(output_path / f"contrast_limits_{id_}.json", "w") as f:
        combined_dict = limits_dict.copy()
        combined_dict["real_limits"] = volume_limit
        json.dump(combined_dict, f, cls=NpEncoder, indent=4)

    # Check which method is closest to the human contrast limits
    closeness_ordering = {
        k: abs(limits_dict[k][0] - volume_limit[0]) + abs(limits_dict[k][1] - volume_limit[1]) for k in limits_dict
    }

    closest_method = min(limits_dict.keys(), key=closeness_ordering.get)
    limits_dict = {k: limits_dict[k] for k in sorted(limits_dict, key=closeness_ordering.get)}
    limits_dict_string = "".join(f"{k} : {v[0]:.5f} - {v[1]:.5f}\n" for k, v in limits_dict.items())
    print(
        f"----------------\nClosest method to human contrast limits of {volume_limit} was {closest_method}:\n{limits_dict_string}Details saved to {output_path}.",
    )
    # For state generation
    limits_dict["closest_method"] = closest_method
    limits_dict["distance_to_human"] = closeness_ordering
    limits_dict["2d"] = volume_limit
    limits_dict["size"] = data_size_dict
    return limits_dict


def run_all_contrast_limit_calculations(
    id_,
    input_data_path,
    output_path,
    z_radius=5,
    num_samples=20000,
):
    output_path.mkdir(parents=True, exist_ok=True)

    human_contrast = id_to_human_contrast_limits[id_]
    volume_limit = human_contrast["volume"]

    # First, percentile contrast limits
    limits_dict = {}
    info_dict = {}
    data = load_omezarr_data(input_data_path, resolution_level=-1, persist=False)
    data_shape = data.shape
    data_size_dict = {"z": data_shape[0], "y": data_shape[1], "x": data_shape[2]}

    # Ensure the main public API is working
    print("GMM from API", compute_contrast_limits(data, method="gmm"))
    print("CDF from API", compute_contrast_limits(data, method="cdf"))

    calculator = ContrastLimitCalculator(
        data,
        z_radius=z_radius,
        num_samples=num_samples,
    )

    # To save doing the downsampling twice
    gmm_calculator = GMMContrastLimitCalculator(calculator.volume)
    cdf_calculator = CDFContrastLimitCalculator(calculator.volume)

    calculator_dict = {
        "gmm": gmm_calculator,
        "cdf": cdf_calculator,
    }
    hyperopt_evals_dict = {
        "gmm": 5,
        "cdf": 100,
    }
    for key, calc in calculator_dict.items():
        max_hyperopt_evals = hyperopt_evals_dict[key]
        best_info, limits = calc.optimize(
            real_limits=volume_limit,
            max_evals=max_hyperopt_evals,
        )
        limits_dict[key] = limits
        info_dict[key] = best_info
        print(
            f"{key}: found ({limits[0]:.4f}, {limits[1]:.4f}) compared to ({volume_limit[0]:.4f}, {volume_limit[1]:.4f}) with:\n{best_info}",
        )
    plot_output_name = output_path / f"contrast_limits_plot_{id_}.png"
    cdf_info = cdf_calculator.calculate_cdf()
    cdf_for_plot = [cdf_info[-1], cdf_info[0]]
    combined_contrast_limit_plot(
        cdf_for_plot,
        human_contrast["volume"],
        limits_dict,
        plot_output_name,
    )

    with open(output_path / f"contrast_limits_{id_}.json", "w") as f:
        combined_dict = {k: {"limits": v, "info": info_dict[k]} for k, v in limits_dict.items()}
        combined_dict["real_limits"] = volume_limit
        combined_dict["estimated_cdf"] = cdf_calculator.automatic_parameter_estimation()
        json.dump(combined_dict, f, cls=NpEncoder, indent=4)

    # Check which method is closest to the human contrast limits
    closeness_ordering = {
        k: abs(limits_dict[k][0] - volume_limit[0]) + abs(limits_dict[k][1] - volume_limit[1]) for k in limits_dict
    }

    closest_method = min(limits_dict.keys(), key=closeness_ordering.get)
    limits_dict = {k: limits_dict[k] for k in sorted(limits_dict, key=closeness_ordering.get)}
    limits_dict_string = "".join(f"{k} : {v[0]:.5f} - {v[1]:.5f}\n" for k, v in limits_dict.items())
    limits_dict["size"] = data_size_dict
    print(
        f"Closest method to human contrast limits of {volume_limit} was {closest_method}:\n{limits_dict_string}Details saved to {output_path}.",
    )
    limits_dict["closest_method"] = closest_method
    limits_dict["distance_to_human"] = closeness_ordering
    limits_dict["2d"] = volume_limit
    return limits_dict


def create_state(id_, contrast_limit_dict, output_folder):
    source, scale = id_to_source_map[id_]
    # One state layer for each contrast limit method
    layers_list = []
    ignored_keys = ["size", "closest_method", "distance_to_human", "2d"]

    contrast_limit_dict["human"] = id_to_human_contrast_limits[id_]["volume"]

    # set the most promising contrast limits as visible, rest not
    for key, limits in contrast_limit_dict.items():
        if key in ignored_keys:
            continue
        visible = key == "gmm"
        closeness = contrast_limit_dict["distance_to_human"].get(key, 0)
        gain = id_to_human_contrast_limits[id_]["gain"] if key == "human" else -7.8
        twodee_limits = contrast_limit_dict["2d"] if key != "human" else id_to_human_contrast_limits[id_]["slice"]
        rounded_closeness = round(closeness, 2)
        layer_info = generate_image_layer(
            source=source,
            scale=scale,
            size=contrast_limit_dict["size"],
            name=f"{id_}_{key}_{rounded_closeness}",
            twodee_contrast_limits=twodee_limits,
            threedee_contrast_limits=limits,
            has_volume_rendering_shader=True,
            volume_rendering_is_visible=True,
            is_visible=visible,
            volume_rendering_gain=gain,
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


def main(
    output_folder,
    take_screenshots=False,
    wait_for=60 * 1000,
    z_radius=5,
    ds_factor=0.3,
):
    url_list = []
    for id_, path in id_to_path_map.items():
        path = Path(output_folder) / path
        grab_tomogram(id_, path)
        limits = run_contrast_limit_calculations_from_api(
            id_,
            path,
            Path(output_folder) / "results",
            z_radius=z_radius,
            ds_factor=ds_factor,
        )
        # For tuning hyperparameters
        # limits = run_all_contrast_limit_calculations(
        #     id_,
        #     path,
        #     Path(output_folder) / "results",
        # )
        state = create_state(id_, limits, Path(output_folder) / "results")
        viewer_state_obj = viewer_state.ViewerState(state)
        url_from_json = to_url(
            viewer_state_obj,
            prefix="https://neuroglancer-demo.appspot.com/",
        )
        url_list.append(url_from_json)
    with open(Path(output_folder) / "urls.txt", "w") as f:
        f.write("\n\n\n".join(url_list))

    if take_screenshots:
        ids = list(id_to_path_map.keys())
        url_dict = {id_: [url] for id_, url in zip(ids, url_list, strict=False)}
        run_screenshot_loop(url_dict, Path(output_folder), wait_for=wait_for)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument("--screenshot", action="store_true")
    parser.add_argument("--wait", default=60, type=int, help="How long to wait before taking the screenshot (in s)")
    args, _ = parser.parse_known_args()
    main(args.output_folder, args.screenshot, wait_for=(args.wait * 1000))
