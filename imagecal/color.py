from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage import color as skcolor


LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float64,
)
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]],
    dtype=np.float64,
)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to display-encoded sRGB without clipping."""
    linear = np.asarray(linear, dtype=np.float32)
    positive = np.maximum(linear, 0.0)
    return np.where(
        positive <= 0.0031308,
        12.92 * positive,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    )


def robust_rgb_median(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)
    return np.median(pixels, axis=0).astype(np.float32)


def neutral_white_balance_gains(gray_rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray_rgb, dtype=np.float64)
    if gray.shape != (3,) or np.any(~np.isfinite(gray)) or np.any(gray <= 0):
        raise ValueError(f"Invalid gray-card RGB values: {gray_rgb!r}")
    target = float(np.exp(np.mean(np.log(gray))))
    return (target / gray).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(rgb, dtype=np.float64) * LUMA_WEIGHTS, axis=-1)


@dataclass(frozen=True)
class Correction:
    gray_rgb: np.ndarray
    wb_gains: np.ndarray
    gray_luminance_after_wb: float
    exposure_gain: float
    delta_ev: float
    combined_gains: np.ndarray


def derive_correction(gray_rgb: np.ndarray, target_luminance: float) -> Correction:
    wb = neutral_white_balance_gains(gray_rgb)
    neutral = np.asarray(gray_rgb, dtype=np.float32) * wb
    y = float(luminance(neutral))
    if target_luminance <= 0 or y <= 0:
        raise ValueError("Gray-card luminance must be positive")
    exposure = float(target_luminance / y)
    return Correction(
        gray_rgb=np.asarray(gray_rgb, dtype=np.float32), wb_gains=wb,
        gray_luminance_after_wb=y, exposure_gain=exposure,
        delta_ev=float(math.log2(exposure)),
        combined_gains=(wb * exposure).astype(np.float32),
    )


def bradford_adaptation(source_white: np.ndarray, target_white: np.ndarray = D65_XYZ) -> np.ndarray:
    source = np.asarray(source_white, dtype=np.float64)
    target = np.asarray(target_white, dtype=np.float64)
    source = source / source[1]
    target = target / target[1]
    cone_source = BRADFORD @ source
    cone_target = BRADFORD @ target
    if np.any(cone_source <= 0) or np.any(~np.isfinite(cone_source)):
        raise ValueError(f"Invalid source white for Bradford adaptation: {source_white!r}")
    return np.linalg.inv(BRADFORD) @ np.diag(cone_target / cone_source) @ BRADFORD


def balanced_camera_to_linear_srgb(rgb_xyz_matrix: np.ndarray, wb_gains: np.ndarray) -> np.ndarray:
    """Build a balanced-camera RGB to D65 linear-sRGB matrix.

    RawPy labels this a camera-RGB/XYZ matrix, but for this D5200 the coefficients
    exactly match the DNG ColorMatrix2.  The DNG specification defines that
    matrix as XYZ to reference-camera native space, so it must be inverted.
    Input pixels have already received ``wb_gains``; the transform accounts for
    that coordinate scaling and chromatically adapts its neutral to D65.
    """
    xyz_to_camera = np.asarray(rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if xyz_to_camera.shape != (3, 3) or abs(np.linalg.det(xyz_to_camera)) < 1e-10:
        raise ValueError("Camera color matrix is missing or singular")
    gains = np.asarray(wb_gains, dtype=np.float64)
    if gains.shape != (3,) or np.any(gains <= 0):
        raise ValueError("White-balance gains must be three positive values")
    balanced_camera_to_xyz = np.linalg.inv(xyz_to_camera) @ np.diag(1.0 / gains)
    source_white = balanced_camera_to_xyz @ np.ones(3, dtype=np.float64)
    scale = float(source_white[1])
    adapted = bradford_adaptation(source_white / scale)
    camera_to_xyz_d65 = adapted @ balanced_camera_to_xyz / scale
    result = XYZ_TO_LINEAR_SRGB @ camera_to_xyz_d65
    return result.astype(np.float32)


def apply_color_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("...c,rc->...r", np.asarray(image, dtype=np.float32), matrix).astype(np.float32)


@dataclass(frozen=True)
class PreviewCurve:
    gray_input: np.ndarray
    white_input: np.ndarray
    gray_output: float
    white_output: float
    white_valid: bool
    fallback_reason: str


def build_preview_curve(
    gray_region: np.ndarray,
    white_region: np.ndarray,
    *,
    gray_code: int = 128,
    white_code: int = 235,
    white_usable: bool = True,
) -> PreviewCurve:
    gray = robust_rgb_median(linear_to_srgb(gray_region))
    white = robust_rgb_median(linear_to_srgb(white_region))
    reason = ""
    valid = bool(white_usable)
    if np.any(~np.isfinite(white)) or np.any(white <= gray * 1.05):
        valid = False
        reason = "white anchor is not safely above gray anchor"
    if not white_usable:
        reason = "white card is sensor-clipped or otherwise invalid"
    return PreviewCurve(
        gray_input=gray, white_input=white,
        gray_output=gray_code / 255.0, white_output=white_code / 255.0,
        white_valid=valid, fallback_reason=reason,
    )


def apply_preview_curve(linear_master: np.ndarray, curve: PreviewCurve) -> np.ndarray:
    """Apply a monotone per-channel display curve; clip only this preview."""
    encoded = linear_to_srgb(linear_master).astype(np.float32)
    result = np.empty_like(encoded)
    for channel in range(3):
        x = np.maximum(encoded[..., channel], 0.0)
        g = max(float(curve.gray_input[channel]), 1e-6)
        yg = curve.gray_output
        if curve.white_valid:
            w = max(float(curve.white_input[channel]), g + 1e-6)
            yw = curve.white_output
            lower = x * (yg / g)
            middle = yg + (x - g) * ((yw - yg) / (w - g))
            upper = yw + (1.0 - yw) * (1.0 - np.exp(-(x - w) / max(w - g, 0.05)))
            result[..., channel] = np.where(x <= g, lower, np.where(x <= w, middle, upper))
        else:
            lower = x * (yg / g)
            upper = yg + (1.0 - yg) * (1.0 - np.exp(-(x - g) / max(g, 0.05)))
            result[..., channel] = np.where(x <= g, lower, upper)
    return np.clip(result, 0.0, 1.0)


def circular_mean_degrees(degrees: np.ndarray) -> tuple[float, float]:
    values = np.asarray(degrees, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    vector = np.mean(np.exp(1j * np.deg2rad(values)))
    return float(np.rad2deg(np.angle(vector)) % 360.0), float(abs(vector))


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_median": float(np.median(arr)), f"{prefix}_std": float(np.std(arr))}


def rgb_patch_lab(linear_rgb: np.ndarray) -> np.ndarray:
    encoded = np.clip(linear_to_srgb(np.asarray(linear_rgb, dtype=np.float32)), 0.0, 1.0)
    return skcolor.rgb2lab(encoded.reshape(1, 1, 3)).reshape(3)


def extract_color_metrics(
    linear_rgb: np.ndarray,
    *, min_hsv_saturation: float = 0.05,
    min_hsv_value: float = 0.02,
    min_lch_chroma: float = 2.0,
) -> dict[str, float | int]:
    # Preserve float-master values for linear RGB; clip only display-derived spaces.
    linear = np.asarray(linear_rgb, dtype=np.float32)
    srgb = np.clip(linear_to_srgb(linear), 0.0, 1.0)
    flat_linear = linear.reshape(-1, 3)
    flat_srgb = srgb.reshape(-1, 3)
    result: dict[str, float | int] = {"pixel_count": int(flat_linear.shape[0])}
    for index, channel in enumerate("RGB"):
        result.update(_stats(f"linear_{channel}", flat_linear[:, index]))
        result.update(_stats(f"srgb_{channel}", flat_srgb[:, index]))
    hsv = skcolor.rgb2hsv(srgb).reshape(-1, 3)
    valid_hue = (hsv[:, 1] >= min_hsv_saturation) & (hsv[:, 2] >= min_hsv_value)
    hue_mean, hue_resultant = circular_mean_degrees(hsv[valid_hue, 0] * 360.0)
    result.update(_stats("HSV_S", hsv[:, 1])); result.update(_stats("HSV_V", hsv[:, 2]))
    result["HSV_H_circular_mean_deg"] = hue_mean
    result["HSV_H_resultant_length"] = hue_resultant
    result["HSV_H_valid_fraction"] = float(np.mean(valid_hue))
    lab = skcolor.rgb2lab(srgb).reshape(-1, 3)
    lch = skcolor.lab2lch(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    for index, channel in enumerate(("L", "a", "b")):
        result.update(_stats(f"Lab_{channel}", lab[:, index]))
    result.update(_stats("LCh_L", lch[:, 0])); result.update(_stats("LCh_C", lch[:, 1]))
    valid_lch = lch[:, 1] >= min_lch_chroma
    lch_mean, lch_resultant = circular_mean_degrees(np.rad2deg(lch[valid_lch, 2]) % 360.0)
    result["LCh_h_circular_mean_deg"] = lch_mean
    result["LCh_h_resultant_length"] = lch_resultant
    result["LCh_h_valid_fraction"] = float(np.mean(valid_lch))
    return result
