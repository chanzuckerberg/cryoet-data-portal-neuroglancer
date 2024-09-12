import argparse
import json
import logging
from pathlib import Path

from cryoet_data_portal import Client, Tomogram
from scipy.spatial.transform import Rotation

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    CDFContrastLimitCalculator,
    ContrastLimitCalculator,
    GMMContrastLimitCalculator,
    KMeansContrastLimitCalculator,
)
from cryoet_data_portal_neuroglancer.state_generator import combine_json_layers, generate_image_layer

# Set up logging - level is info
logging.basicConfig(level=logging.INFO, force=True)

OUTPUT_FOLDER = "/media/starfish/LargeSSD/data/cryoET/data/FromAPI"

id_to_path_map = {
    1000: "1000/16.zarr",
    706: "706/Position_161.zarr",
    800: "800/0105.zarr",
    10845: "10845/ay18112021_grid2_lamella3_position7.zarr",
    4279: "4279/dga2018-08-27-600",
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
        "slice:": [0.0000748111, 0.00189353],
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
}


def grab_tomogram(id_: int, zarr_path: Path):
    client = Client()
    if not zarr_path.exists():
        zarr_path.mkdir(parents=True, exist_ok=True)
        tomogram = Tomogram.get_by_id(client, id_)
        tomogram.download_omezarr(str(zarr_path.parent.resolve()))


def run_all_contrast_limit_calculations(id_, input_data_path, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    # First, percentile contrast limits
    limits_dict = {}
    data = load_omezarr_data(input_data_path)
    data_shape = data.shape
    data_size_dict = {"z": data_shape[0], "y": data_shape[1], "x": data_shape[2]}

    calculator = ContrastLimitCalculator(data)

    # For now, we will trim the volume around the central z-slice
    calculator.trim_volume_around_central_zslice(z_radius=5)
    calculator.take_random_samples_from_volume(20000)

    # Percentile contrast limits
    limits = calculator.contrast_limits_from_percentiles(5.0, 60.0)
    limits_dict["percentile"] = limits

    # K means contrast limits
    kmeans_calculator = KMeansContrastLimitCalculator(calculator.volume)
    limits = kmeans_calculator.contrast_limits_from_kmeans()
    limits_dict["kmeans"] = limits
    kmeans_calculator.plot_kmeans_clusters(output_path / "kmeans_clusters.png")

    # GMM contrast limits
    gmm_calculator = GMMContrastLimitCalculator(calculator.volume)
    limits = gmm_calculator.contrast_limits_from_gmm()
    limits_dict["gmm"] = limits
    gmm_calculator.plot_gmm_clusters(output_path / "gmm_clusters.png")

    # CDF based contrast limits
    cdf_calculator = CDFContrastLimitCalculator(calculator.volume)
    limits = cdf_calculator.contrast_limits_from_cdf()
    limits_dict["cdf"] = limits
    cdf_calculator.plot_cdf(output_path / "cdf.png")

    # 2D contrast limits
    limits = calculator.contrast_limits_from_percentiles(1.0, 99.0)
    limits_dict["2d"] = limits

    with open(output_path / "contrast_limits.json", "w") as f:
        json.dump(limits_dict, f)

    human_contrast = id_to_human_contrast_limits[id_]
    volume_limit = human_contrast["volume"]
    # Check which method is closest to the human contrast limits
    closeness_ordering = {
        k: abs(limits_dict[k][0] - volume_limit[0]) + abs(limits_dict[k][1] - volume_limit[1]) for k in limits_dict
    }

    closest_method = min(limits_dict.keys(), key=closeness_ordering.get)
    limits_dict = {k: limits_dict[k] for k in sorted(limits_dict, key=closeness_ordering.get)}
    limits_dict["size"] = data_size_dict
    print(
        f"Closest method to human contrast limits: {closest_method}, {limits_dict[closest_method]}, human: {volume_limit}. 2D: {limits_dict['2d']}, human {human_contrast['slice']}, Details saved to {output_path}.",
    )
    limits_dict["closest_method"] = closest_method
    limits_dict["distance_to_human"] = closeness_ordering

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
        visible = key == contrast_limit_dict["closest_method"]
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
        layers_list.append(layer_info)
    json_state = combine_json_layers(
        layers_list,
        scale,
        projection_quaternion=Rotation.from_euler(seq="xyz", angles=(0, 0, 0), degrees=True).as_quat(),
    )
    with open(output_folder / f"{id_}_state.json", "w") as f:
        json.dump(json_state, f, indent=4)
    print(f"State file saved to {output_folder / f'{id_}_state.json'}")


def main(output_folder):
    for id_, path in id_to_path_map.items():
        path = Path(output_folder) / path
        grab_tomogram(id_, path)
        limits = run_all_contrast_limit_calculations(
            id_,
            path,
            Path(output_folder) / f"results_{id_}",
        )
        create_state(id_, limits, Path(output_folder))
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    args, _ = parser.parse_known_args()
    main(args.output_folder)
