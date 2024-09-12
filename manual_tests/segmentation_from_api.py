import os

from cloudvolume import CloudVolume
from cryoet_data_portal import Annotation, Client

from cryoet_data_portal_neuroglancer.io import load_omezarr_data
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    clean_mesh_folder,
    generate_multiresolution_mesh_from_segmentation,
    generate_single_resolution_mesh,
)
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import encode_segmentation
from cryoet_data_portal_neuroglancer.utils import determine_size_of_non_zero_bounding_box

# logging.basicConfig(level=logging.DEBUG, force=True)

zarr_path = "102-membrane-1.0_segmentationmask.zarr"
output_path = "102-membrane-1.0_segmentationmask_encoded"
resolution = 1.1  # Non 1.0 is a harder test


def grab_annotation():
    client = Client()
    if not os.path.exists(zarr_path):
        annotation = Annotation.get_by_id(client, 2480)
        annotation.download(format="zarr")


def make_precomputed_segmentation():

    encode_segmentation(
        zarr_path,
        output_path,
        (resolution, resolution, resolution),
        include_mesh=False,
        convert_non_zero_to=1,
        delete_existing=False,
    )


def make_multi_res_mesh():
    dask_data = load_omezarr_data(zarr_path)
    max_simplification_error = 10 * max(1, int(resolution))
    mesh_shape = determine_size_of_non_zero_bounding_box(dask_data)
    del dask_data
    generate_multiresolution_mesh_from_segmentation(
        output_path,
        "mesh",
        max_lod=2,
        mesh_shape=mesh_shape,
        max_simplification_error=max_simplification_error,
    )
    clean_mesh_folder(output_path, "mesh")


def make_single_res_mesh():
    """Likely to fail due to memory constraints"""
    dask_data = load_omezarr_data(zarr_path)
    generate_single_resolution_mesh(
        dask_data,
        output_path,
        "single_mesh",
        (resolution, resolution, resolution),
    )


def serve_files():
    # Multi res mesh
    cv = CloudVolume(f"file://{output_path}")
    cv.viewer(port=1337)

    # Single res mesh
    # cv = CloudVolume(f"file://{output_path}", mesh_dir="single_mesh")
    # cv.viewer(port=1338)


def main():
    grab_annotation()
    make_precomputed_segmentation()
    make_multi_res_mesh()
    # make_single_res_mesh()
    serve_files()


if __name__ == "__main__":
    main()
