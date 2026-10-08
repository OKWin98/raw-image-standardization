from __future__ import annotations

import csv
import copy
import hashlib
import shutil
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
import tifffile
import yaml
from PIL import Image, ImageDraw

from . import __version__
from .color import (
    apply_color_matrix,
    apply_preview_curve,
    balanced_camera_to_linear_srgb,
    build_preview_curve,
    derive_correction,
    extract_color_metrics,
    luminance,
    rgb_patch_lab,
    robust_rgb_median,
)
from .safety import copy_verified, enforce_output_separation, snapshots, verify_snapshots, write_snapshot
from .safety import (
    enforce_disjoint_output,
    prepare_output_directory,
    require_absolute_directory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    local = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    common_name = local.pop("common_config", None)
    if common_name:
        common_path = (path.parent / common_name).resolve()
        common = yaml.safe_load(common_path.read_text(encoding="utf-8")) or {}
        config = _deep_merge(common, local)
        config["_common_config_path"] = str(common_path)
    else:
        config = local
    config["_config_path"] = str(path)
    return config


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Runtime path must be absolute: {path}")
    return path.resolve(strict=False)


def roi_slices(shape: tuple[int, ...], normalized: list[float]) -> tuple[slice, slice]:
    height, width = shape[:2]
    left, top, right, bottom = normalized
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"Invalid normalized ROI: {normalized}")
    x0, x1 = int(round(left * width)), int(round(right * width))
    y0, y1 = int(round(top * height)), int(round(bottom * height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI resolves to an empty region: {normalized}")
    return slice(y0, y1), slice(x0, x1)


def crop_roi(image: np.ndarray, normalized: list[float]) -> np.ndarray:
    ys, xs = roi_slices(image.shape, normalized)
    return image[ys, xs]


def _sample_names(sample: dict[str, Any]) -> tuple[str, str, str | None]:
    stem = sample.get("stem") or Path(sample["raw"]).stem
    jpg_name = sample.get("jpg")
    if "raw" not in sample:
        jpg_name = jpg_name or f"{stem}.jpg"
    return stem, sample.get("raw", f"{stem}.nef"), jpg_name


def source_files(config: dict[str, Any]) -> list[Path]:
    root = Path(config["source_root"])
    result: list[Path] = []
    for sample in config["samples"]:
        _, raw_name, jpg_name = _sample_names(sample)
        for name in (raw_name, jpg_name):
            if name is None:
                continue
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            result.append(path)
    return result


def prepare_samples(config: dict[str, Any]) -> list[dict[str, str | int]]:
    source_root = Path(config["source_root"])
    sample_dir = project_path(config["sample_dir"])
    output_dir = project_path(config["output_dir"])
    enforce_output_separation(source_root, sample_dir, output_dir)
    files = source_files(config)
    before = snapshots(files)
    for source in files:
        copy_verified(source, sample_dir / source.name)
    verify_snapshots(before)
    _write_or_verify_source_snapshot(
        output_dir / "source_snapshot.json",
        before,
        resume=bool(config.get("processing", {}).get("resume", False)),
    )
    return before


def _write_or_verify_source_snapshot(
    path: Path, current: list[dict[str, str | int]], *, resume: bool
) -> None:
    if resume:
        if not path.is_file():
            raise RuntimeError(f"Resume requires an existing source snapshot: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != current:
            raise RuntimeError("Resume refused because source file hashes or metadata changed")
        return
    write_snapshot(path, current)


def _sensor_region_metrics(
    raw_visible: np.ndarray,
    colors_visible: np.ndarray,
    normalized_roi: list[float],
    black_levels: np.ndarray,
    white_level: float,
    saturation_margin: float,
) -> dict[str, Any]:
    ys, xs = roi_slices(raw_visible.shape, normalized_roi)
    values = np.asarray(raw_visible[ys, xs], dtype=np.float64)
    colors = np.asarray(colors_visible[ys, xs], dtype=np.int8)
    saturated = values >= white_level - saturation_margin
    normalized = np.zeros_like(values)
    channel_sat: list[float] = []
    channel_median: list[float] = []
    for color in range(3):
        mask = colors == color
        level = float(black_levels[color])
        normalized[mask] = (values[mask] - level) / max(white_level - level, 1.0)
        channel_sat.append(float(np.mean(saturated[mask])) if np.any(mask) else float("nan"))
        channel_median.append(float(np.median(normalized[mask])) if np.any(mask) else float("nan"))
    # Include the second green code (3) with green if the camera exposes it.
    mask3 = colors == 3
    if np.any(mask3):
        level = float(black_levels[3])
        normalized[mask3] = (values[mask3] - level) / max(white_level - level, 1.0)
        green_values = np.concatenate([normalized[colors == 1], normalized[mask3]])
        green_sat = np.concatenate([saturated[colors == 1], saturated[mask3]])
        channel_median[1] = float(np.median(green_values))
        channel_sat[1] = float(np.mean(green_sat))
    return {
        "saturation_fraction": float(np.mean(saturated)),
        "channel_saturation": channel_sat,
        "channel_median": channel_median,
        "near_black_fraction": float(np.mean(normalized <= 0.001)),
    }


def decode_camera_rgb(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    raw_cfg = config["raw"]
    demosaic = getattr(rawpy.DemosaicAlgorithm, raw_cfg.get("demosaic", "AHD"))
    attempts = int(config.get("processing", {}).get("read_retries", 3))
    for attempt in range(attempts):
        try:
            with rawpy.imread(str(path)) as raw:
                black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
                white = float(raw.white_level)
                sensor = np.array(raw.raw_image_visible, copy=True)
                colors = np.array(raw.raw_colors_visible, copy=True)
                matrix = np.array(raw.rgb_xyz_matrix, dtype=np.float64, copy=True)
                processed = raw.postprocess(
                    output_bps=16, output_color=rawpy.ColorSpace.raw,
                    gamma=(1.0, 1.0), no_auto_bright=True, no_auto_scale=True,
                    use_auto_wb=False, use_camera_wb=False,
                    user_wb=[1.0, 1.0, 1.0, 1.0], demosaic_algorithm=demosaic,
                )
            break
        except rawpy.LibRawIOError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(1.0)
    # With no_auto_scale LibRaw leaves demosaiced values in sensor-count scale,
    # represented in a uint16 container. Black subtraction occurs before demosaic.
    scale = 65535.0 / max(white - float(np.mean(black)), 1.0)
    camera = processed.astype(np.float32) / 65535.0 * scale
    margin = float(raw_cfg.get("sensor_saturation_margin", 16.0))
    meta = {
        "black_levels": black.tolist(), "white_level": white,
        "rgb_xyz_matrix": matrix.tolist(), "normalization_scale": scale,
        "full_sensor_saturation_fraction": float(np.mean(sensor >= white - margin)),
        "gray_sensor": _sensor_region_metrics(sensor, colors, config["rois"]["gray_card"], black, white, margin),
        "white_sensor": _sensor_region_metrics(sensor, colors, config["rois"]["white_card"], black, white, margin),
    }
    return camera, meta


def _patch_uniformity(region: np.ndarray) -> float:
    values = luminance(region)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return mad / max(abs(center), 1e-12)


def _save_preview(display_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(np.round(display_rgb * 255).astype(np.uint8), mode="RGB").save(path, "PNG")


def _save_roi_overlay(jpg_path: Path, rois: dict[str, list[float]], output: Path) -> None:
    image = Image.open(jpg_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"gray_card": "#ffd84d", "white_card": "#40d9ff", "experiment": "#ff4fa3"}
    for name, normalized in rois.items():
        ys, xs = roi_slices((image.height, image.width), normalized)
        box = (xs.start, ys.start, xs.stop, ys.stop)
        width = max(4, image.width // 800)
        draw.rectangle(box, outline=colors.get(name, "#00ff66"), width=width)
        draw.text((box[0] + width, box[1] + width), name, fill=colors.get(name, "#00ff66"))
    image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    image.save(output, "PNG")


def _package_versions() -> dict[str, str]:
    names = ("rawpy", "numpy", "tifffile", "Pillow", "matplotlib", "scikit-image", "PyYAML")
    return {name: importlib.metadata.version(name) for name in names}


def _qc_status(row: dict[str, Any], quality: dict[str, Any]) -> tuple[str, str]:
    reject: list[str] = []
    warn: list[str] = []
    if row["gray_sensor_saturation"] > quality["max_gray_sensor_saturation"]:
        reject.append("gray card has sensor-channel saturation")
    if row["gray_sensor_near_black"] > quality["reject_gray_near_black_fraction"]:
        reject.append("gray card is too close to sensor black")
    if abs(row["delta_ev"]) > quality["warn_ev"]:
        warn.append("exposure correction exceeds EV warning limit")
    if row["white_sensor_saturation"] > quality["max_white_sensor_saturation"]:
        warn.append("white card is sensor-clipped; preview used gray-only shoulder")
    if row["gray_noise_proxy_cv"] > quality["warn_gray_noise_proxy_cv"]:
        warn.append("gray-card noise proxy is elevated")
    if max(abs(row["white_master_astar"]), abs(row["white_master_bstar"])) > quality["warn_white_lab_ab"]:
        warn.append("white-card color residual indicates profile or mixed-light limitation")
    if reject:
        return "REJECT", "; ".join(reject + warn)
    if warn:
        return "WARN", "; ".join(warn)
    return "PASS", ""


def process(config: dict[str, Any], source_before: list[dict[str, str | int]]) -> list[dict[str, Any]]:
    sample_dir = project_path(config["sample_dir"])
    output_dir = project_path(config["output_dir"])
    master_dir, preview_dir, assets = output_dir / "linear_float_tiff", output_dir / "previews", output_dir / "report_assets"
    checkpoint_dir = output_dir / "checkpoints"
    for directory in (master_dir, preview_dir, assets, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    target = float(config["standard"]["gray_reflectance"])
    rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    source_hashes = {Path(str(item["path"])).name: item["sha256"] for item in source_before}
    resume = bool(config.get("processing", {}).get("resume", False))
    make_overlays = bool(config.get("report", {}).get("roi_overlays", True))

    for sample in config["samples"]:
        stem, raw_name, jpg_name = _sample_names(sample)
        master_path = master_dir / f"{stem}_linear_srgb_float32.tif"
        preview_path = preview_dir / f"{stem}_preview.png"
        checkpoint_path = checkpoint_dir / f"{stem}.json"
        if resume and checkpoint_path.is_file() and master_path.is_file() and preview_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("imagecal_version") == __version__ and checkpoint.get("source_sha256") == source_hashes.get(raw_name):
                rows.append(checkpoint["row"]); matrices[raw_name] = checkpoint["matrix_and_raw_metadata"]
                print(f"[{config['camera_id']}] resume {len(rows)}/{len(config['samples'])}: {raw_name}", flush=True)
                continue
        camera, raw_meta = decode_camera_rgb(sample_dir / raw_name, config)
        gray_camera_region = crop_roi(camera, config["rois"]["gray_card"])
        gray_camera = robust_rgb_median(gray_camera_region)
        correction = derive_correction(gray_camera, target)
        balanced = camera * correction.combined_gains.reshape(1, 1, 3)
        matrix = balanced_camera_to_linear_srgb(np.asarray(raw_meta["rgb_xyz_matrix"]), correction.wb_gains)
        master = apply_color_matrix(balanced, matrix)
        del camera, balanced

        gray_master_region = crop_roi(master, config["rois"]["gray_card"])
        white_master_region = crop_roi(master, config["rois"]["white_card"])
        gray_master = robust_rgb_median(gray_master_region)
        white_master = robust_rgb_median(white_master_region)
        gray_lab = rgb_patch_lab(gray_master)
        white_lab = rgb_patch_lab(white_master)
        white_usable = raw_meta["white_sensor"]["saturation_fraction"] <= config["quality"]["max_white_sensor_saturation"]
        curve = build_preview_curve(
            gray_master_region, white_master_region,
            gray_code=int(config["preview"]["gray_code"]),
            white_code=int(config["preview"]["white_code"]), white_usable=white_usable,
        )
        display = apply_preview_curve(master, curve)
        gray_preview = np.round(robust_rgb_median(crop_roi(display, config["rois"]["gray_card"])) * 255)
        white_preview = np.round(robust_rgb_median(crop_roi(display, config["rois"]["white_card"])) * 255)
        maximum = float(np.max(master)); minimum = float(np.min(master))
        above_one = float(np.mean(np.any(master > 1.0, axis=2)))

        gray_noise = _patch_uniformity(gray_master_region)
        row: dict[str, Any] = {
            "camera_id": config["camera_id"], "filename": raw_name,
            "condition": sample["condition"], "capture_order": int(sample["capture_order"]),
            "gray_camera_R": float(gray_camera[0]), "gray_camera_G": float(gray_camera[1]), "gray_camera_B": float(gray_camera[2]),
            "wb_gain_R": float(correction.wb_gains[0]), "wb_gain_G": float(correction.wb_gains[1]), "wb_gain_B": float(correction.wb_gains[2]),
            "exposure_gain": correction.exposure_gain, "delta_ev": correction.delta_ev,
            "gray_master_R": float(gray_master[0]), "gray_master_G": float(gray_master[1]), "gray_master_B": float(gray_master[2]),
            "gray_master_Y": float(luminance(gray_master)),
            "gray_channel_deviation": float(np.max(np.abs(gray_master / np.mean(gray_master) - 1.0))),
            "gray_master_Lstar": float(gray_lab[0]), "gray_master_astar": float(gray_lab[1]), "gray_master_bstar": float(gray_lab[2]),
            "white_master_R": float(white_master[0]), "white_master_G": float(white_master[1]), "white_master_B": float(white_master[2]),
            "white_master_Y": float(luminance(white_master)),
            "white_master_Lstar": float(white_lab[0]), "white_master_astar": float(white_lab[1]), "white_master_bstar": float(white_lab[2]),
            "white_gray_Y_ratio": float(luminance(white_master) / max(luminance(gray_master), 1e-12)),
            "gray_sensor_saturation": raw_meta["gray_sensor"]["saturation_fraction"],
            "white_sensor_saturation": raw_meta["white_sensor"]["saturation_fraction"],
            "full_sensor_saturation": raw_meta["full_sensor_saturation_fraction"],
            "gray_sensor_near_black": raw_meta["gray_sensor"]["near_black_fraction"],
            "gray_noise_proxy_cv": gray_noise, "gray_spatial_uniformity_cv": gray_noise,
            "master_min": minimum, "master_max": maximum, "master_above_one_fraction": above_one,
            "master_dtype": "float32", "master_space": config["master"]["space"],
            "preview_curve_mode": "gray_white_three_point" if curve.white_valid else "gray_plus_shoulder",
            "preview_curve_gray_input_R": float(curve.gray_input[0]), "preview_curve_gray_input_G": float(curve.gray_input[1]), "preview_curve_gray_input_B": float(curve.gray_input[2]),
            "preview_curve_white_input_R": float(curve.white_input[0]), "preview_curve_white_input_G": float(curve.white_input[1]), "preview_curve_white_input_B": float(curve.white_input[2]),
            "preview_gray_R": int(gray_preview[0]), "preview_gray_G": int(gray_preview[1]), "preview_gray_B": int(gray_preview[2]),
            "preview_white_R": int(white_preview[0]), "preview_white_G": int(white_preview[1]), "preview_white_B": int(white_preview[2]),
            "preview_fallback_reason": curve.fallback_reason,
            "color_profile_status": config["color_profile"]["status"],
            "cross_camera_confidence": "neutral_axis_only" if config["color_profile"]["status"] == "neutral_only" else "colorchecker_profiled",
        }
        row["qc_status"], row["qc_reasons"] = _qc_status(row, config["quality"])
        rows.append(row)
        matrices[raw_name] = {"camera_to_linear_srgb": matrix.tolist(), "raw_metadata": raw_meta}

        tifffile.imwrite(
            master_path, master.astype(np.float32),
            photometric="rgb", metadata=None, compression=config["master"].get("compression", "deflate"),
            description="float32 linear sRGB D65; unclipped; gray card normalized to 0.18; neutral-only profile",
        )
        _save_preview(display, preview_path)
        if make_overlays and jpg_name is not None:
            _save_roi_overlay(sample_dir / jpg_name, config["rois"], assets / f"{stem}_roi_overlay.png")
        checkpoint_path.write_text(json.dumps({
            "imagecal_version": __version__, "source_sha256": source_hashes.get(raw_name),
            "row": row, "matrix_and_raw_metadata": matrices[raw_name],
        }, ensure_ascii=False), encoding="utf-8")
        print(f"[{config['camera_id']}] processed {len(rows)}/{len(config['samples'])}: {raw_name}", flush=True)
        del master, display

    log_path = output_dir / "correction_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    manifest = {
        "schema_version": 2, "imagecal_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "packages": _package_versions(), "config": config,
        "source_files": source_before, "per_frame_matrices_and_raw_metadata": matrices,
        "scientific_scope": "Exposure and neutral-axis comparable; absolute color and cross-camera chromatic equivalence are not claimed.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_snapshots(source_before)
    return rows


def extract(config: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = project_path(config["output_dir"]); master_dir = output_dir / "linear_float_tiff"
    settings = config["extraction"]; stride = int(settings["sample_stride"]); rows: list[dict[str, Any]] = []
    for sample in config["samples"]:
        stem, _, _ = _sample_names(sample)
        path = master_dir / f"{stem}_linear_srgb_float32.tif"
        image = tifffile.imread(path)
        roi = crop_roi(image, config["rois"]["experiment"])[::stride, ::stride]
        metrics = extract_color_metrics(
            roi, min_hsv_saturation=float(settings["min_hsv_saturation"]),
            min_hsv_value=float(settings["min_hsv_value"]), min_lch_chroma=float(settings["min_lch_chroma"]),
        )
        rows.append({"camera_id": config["camera_id"], "filename": path.name, "condition": sample["condition"], **metrics})
        del image, roi
    path = output_dir / "color_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    return rows


def validate_acceptance(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    target = float(config["standard"]["gray_reflectance"]); tolerance = float(config["quality"]["gray_target_tolerance"])
    gray_codes = np.array([[r[f"preview_gray_{c}"] for c in "RGB"] for r in rows], dtype=float)
    white_rows = [r for r in rows if r["preview_curve_mode"] == "gray_white_three_point"]
    checks = {
        "gray_master_target": all(abs(r["gray_master_Y"] - target) <= tolerance for r in rows),
        "gray_channel_deviation": all(r["gray_channel_deviation"] <= config["quality"]["max_gray_channel_deviation"] for r in rows),
        "gray_sensor_not_saturated": all(r["gray_sensor_saturation"] <= config["quality"]["max_gray_sensor_saturation"] for r in rows),
        "float_master_unclipped": all(r["master_dtype"] == "float32" for r in rows),
        "preview_gray_target": bool(np.all(np.abs(gray_codes - config["preview"]["gray_code"]) <= config["quality"]["preview_gray_code_tolerance"])),
        "preview_white_target_when_valid": all(
            max(abs(r[f"preview_white_{c}"] - config["preview"]["white_code"]) for c in "RGB") <= config["quality"]["preview_white_code_tolerance"]
            and max(r[f"preview_white_{c}"] for c in "RGB") - min(r[f"preview_white_{c}"] for c in "RGB") <= config["quality"]["preview_channel_spread"]
            for r in white_rows
        ),
        "no_rejected_frames": all(r["qc_status"] != "REJECT" for r in rows),
    }
    by_condition = {r["condition"]: r for r in rows}
    if "midday" in by_condition and "first_night" in by_condition:
        checks["midday_requires_less_gain_than_first_night"] = by_condition["midday"]["delta_ev"] < by_condition["first_night"]["delta_ev"]
    if "second_night" in by_condition and "first_night" in by_condition:
        checks["second_night_requires_more_gain_than_first_night"] = by_condition["second_night"]["delta_ev"] > by_condition["first_night"]["delta_ev"]
    return {"passed": all(checks.values()), "checks": checks, "camera_id": config["camera_id"], "rows": len(rows), "valid_white_anchors": len(white_rows)}


def _runtime_config(
    config_path: Path,
    source_root: Path,
    camera_output_dir: Path,
    *,
    sample_dir: Path,
    resume: bool,
    make_overlays: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    config["source_root"] = str(source_root)
    config["sample_dir"] = str(sample_dir)
    config["output_dir"] = str(camera_output_dir)
    config["processing"] = {"resume": resume, "read_retries": 3}
    config["report"] = {"roi_overlays": make_overlays}
    return config


def build_validation_config(
    config_path: Path, source_root: Path, run_root: Path, *, resume: bool
) -> dict[str, Any]:
    base = load_config(config_path)
    camera_id = base["camera_id"]
    config = _runtime_config(
        config_path,
        source_root,
        run_root / f"camera_{camera_id}",
        sample_dir=run_root / "work" / "validation" / f"camera_{camera_id}",
        resume=resume,
        make_overlays=True,
    )
    config["run_mode"] = "validation"
    return config


def build_extract_config(config_path: Path, run_root: Path) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    config["output_dir"] = str(run_root / f"camera_{config['camera_id']}")
    config["run_mode"] = "validation"
    return config


def _run_validation_config(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    before = prepare_samples(config)
    rows = process(config, before)
    color_rows = extract(config)
    acceptance = validate_acceptance(rows, config)
    output_dir = project_path(config["output_dir"])
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from .report import build_camera_report

    build_camera_report(config, rows, color_rows, acceptance, output_dir)
    verify_snapshots(before)
    return rows, color_rows, acceptance


def run(
    config_path: Path, source_root: Path, run_root: Path, *, resume: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = build_validation_config(config_path, source_root, run_root, resume=resume)
    rows, color_rows, acceptance = _run_validation_config(config)
    return config, rows, color_rows, acceptance


def build_full_series_config(
    config_path: Path, source_root: Path, run_root: Path, *, resume: bool
) -> dict[str, Any]:
    base = load_config(config_path)
    camera_id = base["camera_id"]
    discovery = base["discovery"]
    raw_files = sorted(source_root.glob(discovery["raw_glob"]))
    samples: list[dict[str, Any]] = []
    pattern = re.compile(discovery["filename_regex"], re.IGNORECASE)
    for raw_file in raw_files:
        match = pattern.fullmatch(raw_file.name)
        if not match:
            continue
        samples.append(
            {
                "stem": raw_file.stem,
                "raw": raw_file.name,
                "condition": match.group("capture_time"),
                "capture_order": int(match.group("capture_order")),
            }
        )
    expected = int(discovery["expected_count"])
    if len(samples) != expected:
        raise RuntimeError(
            f"Expected {expected} camera-{camera_id} NEFs matching {discovery['raw_glob']}, "
            f"found {len(samples)}"
        )
    config = _runtime_config(
        config_path,
        source_root,
        run_root / f"camera_{camera_id}",
        sample_dir=source_root,
        resume=resume,
        make_overlays=False,
    )
    config["samples"] = samples
    config["run_mode"] = "full"
    return config


def run_batch(
    config_path: Path, source_root: Path, run_root: Path, *, resume: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = build_full_series_config(config_path, source_root, run_root, resume=resume)
    output_dir = project_path(config["output_dir"])
    enforce_output_separation(source_root, output_dir)
    before = snapshots(source_files(config))
    _write_or_verify_source_snapshot(
        output_dir / "source_snapshot.json", before, resume=resume
    )
    rows = process(config, before)
    color_rows = extract(config)
    acceptance = validate_acceptance(rows, config)
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from .report import build_batch_report

    build_batch_report(config, rows, color_rows, acceptance, output_dir)
    verify_snapshots(before)
    return config, rows, color_rows, acceptance


def dual_camera_config_paths(dual_config_path: Path) -> list[Path]:
    dual_path = dual_config_path.resolve()
    dual_config = yaml.safe_load(dual_path.read_text(encoding="utf-8")) or {}
    return [(dual_path.parent / name).resolve() for name in dual_config["cameras"]]


def _config_fingerprint(paths: list[Path]) -> list[dict[str, str]]:
    fingerprint: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in paths:
        config = load_config(path)
        candidates = [path]
        if common := config.get("_common_config_path"):
            candidates.append(Path(common))
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            fingerprint.append({"name": candidate.name, "sha256": digest})
    return sorted(fingerprint, key=lambda item: item["name"])


def prepare_run_root(
    *,
    command: str,
    config_paths: list[Path],
    source_root: Path,
    output_root: Path,
    resume: bool,
    dry_run: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    source = require_absolute_directory(source_root, "--source-root", must_exist=True)
    output = require_absolute_directory(output_root, "--output-dir", must_exist=False)
    enforce_disjoint_output(output, source, REPOSITORY_ROOT)
    if resume and not output.is_dir():
        raise RuntimeError(f"Cannot resume because the output directory does not exist: {output}")
    prepare_output_directory(output, resume=resume, dry_run=dry_run)
    identity: dict[str, Any] = {
        "schema_version": 1,
        "imagecal_version": __version__,
        "command": command,
        "source_root": str(source),
        "configs": _config_fingerprint(config_paths),
    }
    manifest_path = output / "run_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise RuntimeError(f"Resume requires an existing run manifest: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "imagecal_version", "command", "source_root", "configs"):
            if existing.get(key) != identity[key]:
                raise RuntimeError(f"Resume manifest mismatch for {key}")
    elif not dry_run:
        manifest_path.write_text(
            json.dumps({**identity, "status": "running"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return source, output, identity


def finalize_run_root(output_root: Path, identity: dict[str, Any], acceptance: bool) -> None:
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **identity,
                "status": "complete",
                "acceptance_passed": acceptance,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def preflight(
    config_paths: list[Path], source_root: Path, output_root: Path, *, full: bool
) -> dict[str, Any]:
    cameras: list[dict[str, Any]] = []
    total_uncompressed = 0
    for config_path in config_paths:
        if full:
            config = build_full_series_config(config_path, source_root, output_root, resume=False)
        else:
            config = build_validation_config(config_path, source_root, output_root, resume=False)
            source_files(config)
        raw_paths = [source_root / _sample_names(sample)[1] for sample in config["samples"]]
        first = raw_paths[0]
        with rawpy.imread(str(first)) as raw:
            width, height = int(raw.sizes.width), int(raw.sizes.height)
        uncompressed = len(raw_paths) * width * height * 3 * 4
        total_uncompressed += uncompressed
        cameras.append(
            {
                "camera_id": config["camera_id"],
                "raw_count": len(raw_paths),
                "first_raw": first.name,
                "float32_uncompressed_gib": round(uncompressed / (1024**3), 2),
            }
        )
    free = output_root.anchor and shutil.disk_usage(Path(output_root.anchor)).free
    return {
        "dry_run": True,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "output_will_be_created": not output_root.exists(),
        "cameras": cameras,
        "float32_uncompressed_upper_bound_gib": round(total_uncompressed / (1024**3), 2),
        "output_drive_free_gib": round(free / (1024**3), 2) if free else None,
    }
