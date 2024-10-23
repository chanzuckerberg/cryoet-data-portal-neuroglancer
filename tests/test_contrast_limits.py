import numpy as np

from cryoet_data_portal_neuroglancer.precompute.contrast_limits import (
    PercentileContrastLimitCalculator,
    compute_contrast_limits,
)


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
