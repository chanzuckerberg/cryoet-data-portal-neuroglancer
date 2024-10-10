import os

import cloudvolume
import numpy as np
import pytest
import trimesh
import zarr
from cloudvolume import CloudVolume
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

from cryoet_data_portal_neuroglancer.precompute.mesh import (
    decimate_mesh,
    determine_chunk_size_for_lod,
    generate_mesh_from_lods,
    generate_multilabel_multiresolution_mesh,
    generate_multiresolution_mesh,
)
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import encode_segmentation


@pytest.mark.parametrize(
    "mesh_shape, max_lod, min_chunk_dim, expected_result, expected_lods",
    [
        ((256, 128, 512), 3, 1, ((32, 16, 64), (256 // 8, 128 // 8, 512 // 8)), 4),  # Result: (32, 16, 64)
        ((120, 20, 42), 4, 64, ((120, 20, 42), (64, 16, 32)), 1),  # Can't respect num_lod or min_chunk_dim
        ((60, 40, 139), 4, 16, ((30, 20, 70), (16, 16, 64)), 2),  # Can't respect num_lod
        ((512, 345, 415), 4, 4, ((32, 22, 26), (32, 16, 16)), 5),  # Can respect all
        ((256, 256, 256), 3, 256, ((256, 256, 256), (256, 256, 256)), 1),  # Can respect min_chunk_dim only
    ],
)
def test_determine_chunk_size_for_lod(mesh_shape, max_lod, min_chunk_dim, expected_result, expected_lods):
    actual_expected_result, min_expected_result = expected_result
    (actual_chunk_size, min_chunk_size), num_lods = determine_chunk_size_for_lod(mesh_shape, max_lod, min_chunk_dim)

    assert actual_chunk_size == actual_expected_result
    assert min_chunk_size == min_expected_result
    assert num_lods == expected_lods


def generic_mesh_checks(tmp_path, num_files=2):
    # There should be a mesh folder and an info file
    assert (tmp_path / "mesh").is_dir()
    assert (tmp_path / "info").is_file()
    # Nothing else should be present in the output directory
    assert len(list(tmp_path.iterdir())) == num_files

    # In the mesh folder, there should be also an info file
    assert (tmp_path / "mesh" / "info").is_file()

    # There should either be one shard file
    assert (tmp_path / "mesh" / "0.shard").is_file()
    # Nothing else should be present in the mesh directory
    assert len(list((tmp_path / "mesh").iterdir())) == 2

    # We can load the mesh back
    cv = CloudVolume(f"file://{tmp_path.resolve()}")
    assert isinstance(cv.mesh, cloudvolume.datasource.precomputed.mesh.multilod.ShardedMultiLevelPrecomputedMeshSource)


def test_generate_multiresolution_mesh(tmp_path):
    torus = trimesh.creation.torus(major_radius=5.0, minor_radius=2.5)
    scene = trimesh.Scene(torus)
    generate_multiresolution_mesh(
        scene,
        tmp_path,
        max_lod=2,
        min_mesh_chunk_dim=2,
        bounding_box_size=[100, 100, 120],
    )
    generic_mesh_checks(tmp_path)


def test_generate_mesh_from_lods(tmp_path):
    torus = trimesh.creation.torus(major_radius=5.0, minor_radius=2.5)
    lods = [trimesh.Scene(mesh) for mesh in decimate_mesh(torus, 3, as_trimesh=True)]
    generate_mesh_from_lods(lods, tmp_path, min_mesh_chunk_dim=2)
    generic_mesh_checks(tmp_path)


def test_generate_multilabel_multiresolution_mesh(tmp_path):
    torus = trimesh.creation.torus(major_radius=5.0, minor_radius=2.5)
    scene = trimesh.Scene(torus)
    box = trimesh.creation.box([5, 5, 5])
    scene2 = trimesh.Scene(box)
    labels = {1: scene, 2: scene2}
    generate_multilabel_multiresolution_mesh(
        labels,
        tmp_path,
        max_lod=2,
        min_mesh_chunk_dim=2,
        bounding_box_size=[100, 100, 120],
    )
    generic_mesh_checks(tmp_path)


@pytest.mark.parametrize(
    "segmentation_shape",
    [(128, 128, 128), (120, 53, 12), (40, 20, 25)],
)
def test_mesh_from_segmentation(segmentation_shape, tmp_path):
    zarr_path = tmp_path / "segmentation.zarr"
    output_path = tmp_path / "converted"

    # Generate OME-zarr data and write it to tmp_path
    segmentation_shape = segmentation_shape[::-1]  # Z, Y, X
    data = np.zeros(segmentation_shape, dtype=np.uint8)
    data[:, :10, :] = 1
    data[:, :, :10] = 2

    # write the image data
    if not zarr_path.exists():
        store = parse_url(zarr_path, mode="w").store
        root = zarr.group(store=store)
        write_image(image=data, group=root, axes="zyx")

    encode_segmentation(
        zarr_path,
        output_path,
        resolution=[1, 1, 1],
        include_mesh=True,
        convert_non_zero_to=1,
        max_lod=2,
        delete_existing=True,
    )

    # A segmentation should be created
    assert (output_path / "data").is_dir()

    # Print the contents of the mesh directory
    print(os.listdir(output_path / "mesh"))

    # A mesh shouuld be created
    # generic_mesh_checks(output_path, num_files=4)


if __name__ == "__main__":
    from pathlib import Path

    test_mesh_from_segmentation((120, 53, 12), Path("./tmp"))
