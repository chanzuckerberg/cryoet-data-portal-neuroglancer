import cloudvolume
import pytest
import trimesh
from cloudvolume import CloudVolume

from cryoet_data_portal_neuroglancer.precompute.mesh import determine_chunk_size_for_lod, generate_multiresolution_mesh


@pytest.mark.parametrize(
    "mesh_shape, num_lod, min_chunk_dim, expected_result",
    [
        ((256, 128, 512), 3, 1, (256 / 8, 128 / 8, 512 / 8)),  # Result: (32, 16, 64)
        ((120, 20, 42), 4, 64, (64, 16, 32)),  # Can't respect num_lod or min_chunk_dim
        ((60, 40, 139), 4, 16, (16, 16, 64)),  # Can't respect num_lod
        ((512, 345, 415), 4, 4, (32, 16, 16)),  # Can respect all
        ((256, 256, 256), 3, 256, (256, 256, 256)),  # Can respect min_chunk_dim only
    ],
)
def test_determine_chunk_size_for_lod(mesh_shape, num_lod, min_chunk_dim, expected_result):
    assert determine_chunk_size_for_lod(mesh_shape, num_lod, min_chunk_dim) == expected_result


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

    # There should be a mesh folder and an info file
    assert (tmp_path / "mesh").is_dir()
    assert (tmp_path / "info").is_file()
    # Nothing else should be present in the output directory
    assert len(list(tmp_path.iterdir())) == 2

    # In the mesh folder, there should be also an info file
    # And one shard file
    assert (tmp_path / "mesh" / "info").is_file()
    assert (tmp_path / "mesh" / "0.shard").is_file()
    # Nothing else should be present in the mesh directory
    assert len(list((tmp_path / "mesh").iterdir())) == 2

    # We can load the mesh back
    cv = CloudVolume(f"file://{tmp_path.resolve()}")
    assert isinstance(cv.mesh, cloudvolume.datasource.precomputed.mesh.multilod.ShardedMultiLevelPrecomputedMeshSource)
