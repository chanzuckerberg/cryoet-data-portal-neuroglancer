import pytest

from cryoet_data_portal_neuroglancer.precompute.mesh import determine_chunk_size_for_lod


@pytest.mark.parametrize(
    "mesh_shape, num_lod, min_chunk_dim, expected_result",
    [
        ((256, 128, 512), 3, 1, (256 / 8, 128  / 8, 512 / 8)), # Result: (32, 16, 64)
        ((120, 20, 42), 4, 64, (64, 16, 32)), # Can't respect num_lod or min_chunk_dim
        ((60, 40, 139), 4, 16, (16, 16, 64)), # Can't respect num_lod
        ((512, 345, 415), 4, 4, (32, 16, 16)), # Can respect all
        ((256, 256, 256), 3, 256, (256, 256, 256)), # Can respect min_chunk_dim only
    ],
)
def test_determine_chunk_size_for_lod(mesh_shape, num_lod, min_chunk_dim, expected_result):
    assert determine_chunk_size_for_lod(mesh_shape, num_lod, min_chunk_dim) == expected_result
