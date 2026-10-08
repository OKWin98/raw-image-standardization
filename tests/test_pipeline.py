from pathlib import Path

import numpy as np
import pytest

from imagecal.pipeline import roi_slices, validate_acceptance
from imagecal.safety import enforce_output_separation


def test_normalized_roi_to_slices():
    ys, xs = roi_slices((100, 200, 3), [0.25, 0.2, 0.75, 0.8])
    assert (ys.start, ys.stop, xs.start, xs.stop) == (20, 80, 50, 150)


def test_output_inside_source_is_rejected(tmp_path: Path):
    with pytest.raises(RuntimeError):
        enforce_output_separation(tmp_path, tmp_path / "output")


def test_acceptance_checks_expected_ev_signs():
    rows = [
        {"condition": "first_night", "delta_ev": 0.0},
        {"condition": "midday", "delta_ev": -1.0},
        {"condition": "second_night", "delta_ev": 2.0},
    ]
    for row in rows:
        row.update({"gray_master_Y": 0.18, "gray_channel_deviation": 0.0,
                    "gray_sensor_saturation": 0.0, "master_dtype": "float32",
                    "preview_curve_mode": "gray_white_three_point",
                    "qc_status": "PASS",
                    "preview_gray_R": 128, "preview_gray_G": 128, "preview_gray_B": 128,
                    "preview_white_R": 235, "preview_white_G": 235, "preview_white_B": 235})
    config = {
        "camera_id": "A",
        "standard": {"gray_reflectance": 0.18},
        "preview": {"gray_code": 128, "white_code": 235},
        "quality": {
            "gray_target_tolerance": 0.003,
            "max_gray_channel_deviation": 0.01,
            "max_gray_sensor_saturation": 0.0,
            "preview_gray_code_tolerance": 3,
            "preview_white_code_tolerance": 5,
            "preview_channel_spread": 3,
        }
    }
    result = validate_acceptance(rows, config)
    assert result["passed"]

