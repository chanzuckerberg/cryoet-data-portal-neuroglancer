import numpy as np
import pytest

from cryoet_data_portal_neuroglancer.io import dask_chunks, decode_cf, load_omezarr_data, open_omezarr
from tests.conftest import write_ome_zarr


@pytest.fixture
def volume() -> np.ndarray:
    return np.arange(16 * 20 * 24, dtype=np.float32).reshape(16, 20, 24)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_open_omezarr_v2_and_v3(volume, version, tmp_path):
    path = write_ome_zarr(tmp_path / "image.zarr", volume, version=version, chunks=(8, 8, 8), levels=3, voxel_size=2.5)
    image = open_omezarr(path)
    assert image.zarr_format == (3 if version == "0.5" else 2)
    assert image.ngff_version == version
    assert image.axes == ["z", "y", "x"]
    assert [d["path"] for d in image.datasets] == ["0", "1", "2"]
    assert image.scale(2) == [10.0, 10.0, 10.0]
    assert image.translation(1) == [0.0, 0.0, 0.0]
    assert image.array(-1).shape == (4, 5, 6)
    assert image.array("1").shape == (8, 10, 12)
    np.testing.assert_array_equal(load_omezarr_data(path, persist=False).compute(), volume)
    np.testing.assert_array_equal(load_omezarr_data(path, resolution_level=-1).compute(), volume[::4, ::4, ::4])


def test_load_5d_volume_squeezes_t_and_c(volume, tmp_path):
    path = write_ome_zarr(
        tmp_path / "image.zarr",
        volume[None, None],
        axes=["t", "c", "z", "y", "x"],
        chunks=(1, 1, 8, 8, 8),
    )
    data = load_omezarr_data(path)
    assert data.shape == volume.shape
    np.testing.assert_array_equal(data.compute(), volume)
    assert load_omezarr_data(path, squeeze=False).shape == (1, 1, *volume.shape)


def test_sharded_array_gets_shard_aligned_dask_chunks(tmp_path):
    data = np.zeros((1, 1, 300, 600, 600), dtype=np.int16)
    path = write_ome_zarr(
        tmp_path / "image.zarr",
        data,
        axes=["t", "c", "z", "y", "x"],
        chunks=(1, 1, 128, 128, 128),
        shards=(1, 1, 384, 640, 640),
    )
    image = open_omezarr(path)
    assert dask_chunks(image.array(0)) == (1, 1, 256, 256, 256)
    assert load_omezarr_data(path, persist=False, squeeze=False).chunksize == (1, 1, 256, 256, 256)
    assert load_omezarr_data(path, persist=False, chunks="native", squeeze=False).chunksize == (1, 1, 128, 128, 128)


def test_cf_decoding(tmp_path):
    codes = np.array([[[-32767, 0, 32767]]], dtype=np.int16)
    attrs = {"scale_factor": 2.0e-10, "add_offset": -8.0e-8}
    path = write_ome_zarr(tmp_path / "image.zarr", codes, attributes=attrs)
    image = open_omezarr(path)
    assert image.scale_factor == attrs["scale_factor"]
    assert image.add_offset == attrs["add_offset"]
    np.testing.assert_array_equal(load_omezarr_data(path).compute(), codes)  # stored values by default
    decoded = load_omezarr_data(path, decode=True).compute()
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, attrs["add_offset"] + attrs["scale_factor"] * codes, rtol=1e-6)
    assert decode_cf(codes, None, None) is codes


@pytest.mark.parametrize(
    "version, location",
    [("0.5", "ome"), ("0.4", "ome"), ("0.5", "multiscales-metadata"), ("0.4", "multiscales-metadata")],
)
def test_label_values(version, location, tmp_path):
    labels = np.array([[[0, 1, 5]]], dtype=np.uint16)
    path = write_ome_zarr(
        tmp_path / "labels.zarr",
        labels,
        version=version,
        label_values=[1, 5],
        label_location=location,
    )
    assert open_omezarr(path).label_values == [1, 5]


def test_no_label_values(volume, tmp_path):
    assert open_omezarr(write_ome_zarr(tmp_path / "image.zarr", volume)).label_values is None
