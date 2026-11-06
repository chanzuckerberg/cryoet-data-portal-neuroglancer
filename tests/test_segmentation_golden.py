"""encode_segmentation must write byte-identical output to the last release on zarr 2 / numpy 1
(tests/golden/segmentation_v1.7.2.json), whether the input is Zarr v2 / OME-NGFF 0.4 or Zarr v3 / OME-NGFF 0.5."""

import json
from pathlib import Path

import pytest

from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import encode_segmentation
from tests.conftest import write_ome_zarr
from tests.golden.make_segmentation_golden import CASES, case_data, output_hashes

GOLDEN = json.loads((Path(__file__).parent / "golden" / "segmentation_v1.7.2.json").read_text())


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_encode_segmentation_matches_release(name, version, tmp_path):
    _, chunks, kwargs = CASES[name]
    src = write_ome_zarr(tmp_path / "input.zarr", case_data(name), version=version, chunks=chunks)
    out = tmp_path / "out"
    encode_segmentation(str(src), out, resolution=(1.0, 1.0, 1.0), **kwargs)
    assert output_hashes(out) == GOLDEN[name]


def test_encode_segmentation_sharded_v3_input(tmp_path):
    """A sharded v3 input with small inner chunks gives the same segmentation data as the unsharded input with
    chunks equal to the dask blocks the reader picks for it."""
    name = "labels_default"
    _, _, kwargs = CASES[name]
    data = case_data(name)
    sharded = write_ome_zarr(tmp_path / "sharded.zarr", data, chunks=(8, 8, 8), shards=(32, 32, 32))
    plain = write_ome_zarr(tmp_path / "plain.zarr", data, chunks=(32, 32, 32))
    encode_segmentation(str(sharded), tmp_path / "a", resolution=(1.0, 1.0, 1.0), **kwargs)
    encode_segmentation(str(plain), tmp_path / "b", resolution=(1.0, 1.0, 1.0), **kwargs)
    assert output_hashes(tmp_path / "a") == output_hashes(tmp_path / "b")
