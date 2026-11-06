import numpy as np

from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    PercentileContrastLimitCalculator,
    code_to_physical,
    compute_contrast_limits,
    contrast_limits_from_statistics,
    physical_to_code,
)


def test_physical_to_code_round_trip():
    sf, ao = 2.039e-10, -8.319e-08
    for v in (-6.763e-06, 0.0, 3.128e-06):
        assert np.isclose(code_to_physical(physical_to_code(v, sf, ao), sf, ao), v, rtol=1e-12, atol=0)
    assert physical_to_code(ao, sf, ao) == 0.0
    assert physical_to_code(5.0, None, None) == 5.0


def test_contrast_limits_from_statistics():
    stats = {"p1": -6210.0, "p5": -4352.0, "p95": 4296.0, "p99": 6180.0, "mean": 0.4, "std": 2647.0}
    assert contrast_limits_from_statistics(stats) == (0.4 - 3 * 2647.0, 0.4 + 3 * 2647.0)
    assert contrast_limits_from_statistics(stats, method="percentile") == (-6210.0, 6180.0)
    assert contrast_limits_from_statistics(stats, method="percentile", percentiles=("p5", "p95")) == (-4352.0, 4296.0)


def test_compute_contrast_limits_squeezes_5d():
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(40, 32, 32)).astype(np.float32)
    assert compute_contrast_limits(volume[None, None], method="cdf") == compute_contrast_limits(volume, method="cdf")


def test_percentile_contrast_limits():
    data = np.arange(100).reshape(10, 5, 2)

    calculator = PercentileContrastLimitCalculator(data)
    limits = calculator.compute_contrast_limit(5.0, 95.0)
    assert np.allclose(limits, (5.0, 94.0), atol=0.1)

    limits = calculator.compute_contrast_limit(0.0, 100.0)
    assert limits == (0.0, 99.0)

    # Test with a 3D numpy array with Z as the first axis, and remove some slices
    new_data = np.full((30, 5, 2), 500)
    new_data[10:20] = data

    calculator = PercentileContrastLimitCalculator(new_data)
    limits = calculator.compute_contrast_limit(0.0, 100.0)
    assert limits == (0.0, 500.0)

    calculator = PercentileContrastLimitCalculator(new_data, z_radius=5)
    limits = calculator.compute_contrast_limit(0.0, 100.0)
    assert limits == (0.0, 99.0)


def test_api_contrast_limits():
    """Test the main entry point for computing contrast limits.

    This test is to ensure that some values are returned and that the limits are in the correct order."""
    np.random.seed(42)
    data = np.random.rand(10, 5, 2)
    gmm_limits = compute_contrast_limits(data, "gmm")
    assert isinstance(gmm_limits, tuple)
    assert gmm_limits[0] < gmm_limits[1]

    cdf_limits = compute_contrast_limits(data, "cdf")
    assert isinstance(cdf_limits, tuple)
    assert cdf_limits[0] < cdf_limits[1]


# This is a long test, so it is disabled by default
# def test_hyperparameter_optimization():
#     data = np.arange(100).reshape(10, 5, 2)
#     calculator = ContrastLimitCalculator(data)
#     real_limits = (5, 95)

#     result = calculator.optimize(real_limits, max_evals=2500, loss_threshold=0.1)
#     assert result["low_percentile"] == 5
#     assert result["high_percentile"] == 96
