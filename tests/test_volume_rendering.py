import numpy as np

from cryoet_data_portal_neuroglancer.precompute.volume_rendering import volume_render


def _render(volume: np.ndarray) -> np.ndarray:
    shm, shape = volume_render(volume, (-500.0, 500.0), depth_samples=8, num_workers=1)
    try:
        return np.ndarray(shape, dtype=np.float32, buffer=shm.buf).copy()
    finally:
        shm.close()
        shm.unlink()


def test_volume_render_accepts_int16_and_5d():
    rng = np.random.default_rng(0)
    codes = rng.integers(-1000, 1000, size=(8, 6, 5), dtype=np.int16)
    reference = _render(codes.astype(np.float32))
    assert np.any(reference != 0)
    np.testing.assert_array_equal(_render(codes), reference)
    np.testing.assert_array_equal(_render(codes[None, None]), reference)
