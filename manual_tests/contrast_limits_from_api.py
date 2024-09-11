import argparse
import json
import logging
from pathlib import Path

from cloudvolume import CloudVolume
from cryoet_data_portal import Client, Tomogram

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    CDFContrastLimitCalculator,
    ContrastLimitCalculator,
    GMMContrastLimitCalculator,
    KMeansContrastLimitCalculator,
)

# Set up logging - level is info
logging.basicConfig(level=logging.INFO, force=True)

OUTPUT_FOLDER = "/media/starfish/LargeSSD/data/cryoET/data/FromAPI"

id_to_path_map = {
    1000: "1000/16.zarr",
}

id_to_human_contrast_limits = {
    1000: {
        "slice": [-0.2, 0.15],
        "volume": [-0.035, 0.009],
        "gain": -7.6,
    },
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

    calculator = ContrastLimitCalculator(data)

    # For now, we will trim the volume around the central z-slice
    calculator.trim_volume_around_central_zslice()

    # Percentile contrast limits
    limits = calculator.contrast_limits_from_percentiles(5.0, 80.0)
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

    print(limits_dict)
    with open(output_path / "contrast_limits.json", "w") as f:
        json.dump(limits_dict, f)

    human_contrast = id_to_human_contrast_limits[id_]
    volume_limit = human_contrast["volume"]
    # Check which method is closest to the human contrast limits
    closest_method = min(
        limits_dict.keys(),
        key=lambda x: abs(limits_dict[x][0] - volume_limit[0]) + abs(limits_dict[x][1] - volume_limit[1]),
    )
    print(
        f"Closest method to human contrast limits: {closest_method}, {limits_dict[closest_method]}, human: {volume_limit}. Details saved to {output_path}.",
    )

    return limits_dict


def serve_files(output_path="output.zarr"):
    # Multi res mesh
    cv = CloudVolume(f"file://{output_path}")
    cv.viewer(port=1337)


def main(output_folder):
    for id_, path in id_to_path_map.items():
        path = Path(output_folder) / path
        grab_tomogram(id_, path)
        run_all_contrast_limit_calculations(
            id_,
            path,
            Path(output_folder) / f"results_{id_}",
        )
    # serve_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    args, _ = parser.parse_known_args()
    main(args.output_folder)
