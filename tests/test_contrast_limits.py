import numpy as np

from cryoet_data_portal_neuroglancer.precompute.contrast_limits import ContrastLimitCalculator


def test_percentile_contrast_limits():
    data = np.arange(100).reshape(10, 5, 2)

    calculator = ContrastLimitCalculator(data)
    limits = calculator.compute_contrast_limit(5.0, 95.0)
    assert np.allclose(limits, (5.0, 94.0), atol=0.1)

    limits = calculator.compute_contrast_limit(0.0, 100.0)
    assert limits == (0.0, 99.0)

    # Test with a 3D numpy array with Z as the first axis, and remove some slices
    new_data = np.full((30, 5, 2), 500)
    new_data[10:20] = data

    calculator.volume = new_data
    limits = calculator.compute_contrast_limit(0.0, 100.0)
    assert limits == (0.0, 500.0)

    calculator.set_volume_and_z_limits(new_data, z_radius=5)
    limits = calculator.compute_contrast_limit(0.0, 100.0)
    assert limits == (0.0, 99.0)


def test_mean_rms_contrast_limits():
    data = np.arange(100).reshape(10, 5, 2)

    calculator = ContrastLimitCalculator(data)
    limits = calculator.contrast_limits_from_mean(multipler=1.0)
    # Mean 49.5, RMS 57.3
    assert np.allclose(limits, (-7.8, 106.8), atol=0.1)
