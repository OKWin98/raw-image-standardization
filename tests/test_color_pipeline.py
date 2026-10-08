import numpy as np

from imagecal.color import (
    PreviewCurve,
    apply_color_matrix,
    apply_preview_curve,
    balanced_camera_to_linear_srgb,
    extract_color_metrics,
)


def test_camera_matrix_keeps_balanced_neutral_at_d65_srgb():
    rawpy_matrix = np.array([
        [0.8322, -0.3112, -0.1047],
        [-0.6367, 1.4342, 0.2179],
        [-0.0988, 0.1638, 0.6394],
        [0.0, 0.0, 0.0],
    ])
    wb = np.array([1.2, 0.9, 1.1])
    matrix = balanced_camera_to_linear_srgb(rawpy_matrix, wb)
    # Regression guard: the opposite orientation gives ~[7.87, -4.90, -1.97].
    assert np.allclose(matrix[0], [1.785872, -0.903024, 0.117152], atol=2e-5)
    value = apply_color_matrix(np.array([[[0.18, 0.18, 0.18]]], dtype=np.float32), matrix)[0, 0]
    assert np.allclose(value, [0.18, 0.18, 0.18], atol=2e-5)


def test_preview_three_point_hits_gray_and_white_codes():
    encoded_gray = 0.42
    encoded_white = 0.86
    def inv_srgb(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    image = np.array([[[inv_srgb(encoded_gray)] * 3, [inv_srgb(encoded_white)] * 3]], dtype=np.float32)
    curve = PreviewCurve(np.array([encoded_gray] * 3), np.array([encoded_white] * 3), 128/255, 235/255, True, "")
    result = np.round(apply_preview_curve(image, curve) * 255)
    assert np.all(result[0, 0] == 128)
    assert np.all(result[0, 1] == 235)


def test_float_linear_metrics_do_not_clip_values_above_one():
    image = np.array([[[1.5, 0.2, -0.1]]], dtype=np.float32)
    metrics = extract_color_metrics(image)
    assert metrics["linear_R_mean"] == 1.5
    assert np.isclose(metrics["linear_B_mean"], -0.1)
