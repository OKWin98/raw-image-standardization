import numpy as np

from imagecal.color import (
    circular_mean_degrees,
    derive_correction,
    extract_color_metrics,
    linear_to_srgb,
)


def test_linear_to_srgb_endpoints():
    values = linear_to_srgb(np.array([0.0, 0.0031308, 1.0], dtype=np.float32))
    assert np.allclose(values, [0.0, 0.0404499, 1.0], atol=1e-5)


def test_known_cast_and_exposure_are_corrected():
    gray = np.array([0.08, 0.10, 0.20], dtype=np.float32)
    correction = derive_correction(gray, target_luminance=0.25)
    corrected = gray * correction.combined_gains
    assert np.allclose(corrected, [0.25, 0.25, 0.25], atol=1e-6)


def test_hue_wraparound_uses_circular_mean():
    mean, concentration = circular_mean_degrees(np.array([359.0, 1.0]))
    assert mean < 1.1 or mean > 358.9
    assert concentration > 0.99


def test_low_saturation_hue_is_excluded():
    neutral = np.full((4, 4, 3), 0.25, dtype=np.float32)
    metrics = extract_color_metrics(neutral, min_hsv_saturation=0.05)
    assert metrics["HSV_H_valid_fraction"] == 0.0
    assert np.isnan(metrics["HSV_H_circular_mean_deg"])

