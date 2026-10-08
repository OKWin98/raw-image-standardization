from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(value: Any, digits: int = 5) -> str:
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def _trend_plot(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    ordered = sorted(rows, key=lambda r: (r["camera_id"], r["capture_order"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for camera in sorted({r["camera_id"] for r in ordered}):
        selected = [r for r in ordered if r["camera_id"] == camera]
        x = [r["capture_order"] for r in selected]
        axes[0, 0].plot(x, [r["delta_ev"] for r in selected], "o-", label=f"Camera {camera}")
        for channel, style in zip("RGB", ("r-", "g-", "b-")):
            axes[0, 1].plot(x, [r[f"wb_gain_{channel}"] for r in selected], style, marker="o", alpha=.8, label=f"{camera}-{channel}")
        axes[1, 0].plot(x, [r["white_master_astar"] for r in selected], "o-", label=f"{camera} a*")
        axes[1, 0].plot(x, [r["white_master_bstar"] for r in selected], "s--", label=f"{camera} b*")
        axes[1, 1].plot(x, [r["gray_noise_proxy_cv"] for r in selected], "o-", label=f"Camera {camera}")
    for ax, name, ylabel in zip(axes.flat, ("Exposure correction", "Camera-space WB gains", "White-card residual", "Gray-card noise / nonuniformity"), ("Delta EV", "gain", "Lab a* / b*", "robust CV")):
        ax.set_title(name); ax.set_xlabel("capture order"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(output, dpi=150); plt.close(fig)


def build_camera_report(config: dict[str, Any], rows: list[dict[str, Any]], color_rows: list[dict[str, Any]], acceptance: dict[str, Any], output_dir: Path) -> None:
    assets = output_dir / "report_assets"; trend = assets / "quality_trends.png"
    _trend_plot(rows, trend, f"Camera {config['camera_id']} validation trends")
    color_by_stem = {Path(r["filename"]).name.replace("_linear_srgb_float32.tif", ""): r for r in color_rows}
    cards = []
    for row in sorted(rows, key=lambda r: r["capture_order"]):
        stem = Path(row["filename"]).stem; color = color_by_stem.get(stem, {})
        cards.append(f"""
        <section class="card"><h2>{html.escape(row['condition'])} · {html.escape(row['filename'])}</h2>
        <div class="images"><figure><img src="report_assets/{stem}_roi_overlay.png"><figcaption>原始 JPG 与固定 ROI</figcaption></figure>
        <figure><img src="previews/{stem}_preview.png"><figcaption>统一显示预览（不用于定量）</figcaption></figure></div>
        <table>
        <tr><th>QC</th><td class="{row['qc_status'].lower()}">{row['qc_status']}</td><th>Delta EV</th><td>{_fmt(row['delta_ev'])}</td></tr>
        <tr><th>母版灰卡 RGB / Y</th><td colspan="3">{_fmt(row['gray_master_R'])} / {_fmt(row['gray_master_G'])} / {_fmt(row['gray_master_B'])} · Y={_fmt(row['gray_master_Y'])}</td></tr>
        <tr><th>预览灰卡 / 白卡 code</th><td colspan="3">{row['preview_gray_R']}/{row['preview_gray_G']}/{row['preview_gray_B']} · {row['preview_white_R']}/{row['preview_white_G']}/{row['preview_white_B']}</td></tr>
        <tr><th>WB gains R/G/B</th><td colspan="3">{_fmt(row['wb_gain_R'])} / {_fmt(row['wb_gain_G'])} / {_fmt(row['wb_gain_B'])}</td></tr>
        <tr><th>白卡 Lab / 白灰 Y 比</th><td colspan="3">L*={_fmt(row['white_master_Lstar'])}, a*={_fmt(row['white_master_astar'])}, b*={_fmt(row['white_master_bstar'])} · ratio={_fmt(row['white_gray_Y_ratio'])}</td></tr>
        <tr><th>传感器饱和率</th><td colspan="3">gray={_fmt(row['gray_sensor_saturation'])}, white={_fmt(row['white_sensor_saturation'])}, full={_fmt(row['full_sensor_saturation'])}</td></tr>
        <tr><th>母版范围 / &gt;1 比例</th><td colspan="3">{_fmt(row['master_min'])} … {_fmt(row['master_max'])} · {_fmt(row['master_above_one_fraction'])}</td></tr>
        <tr><th>预览曲线</th><td>{row['preview_curve_mode']}</td><th>实验区 Hue</th><td>{_fmt(color.get('HSV_H_circular_mean_deg', float('nan')))}° (R={_fmt(color.get('HSV_H_resultant_length', float('nan')))})</td></tr>
        <tr><th>说明</th><td colspan="3">{html.escape(row['qc_reasons'] or '无异常')}</td></tr></table></section>""")
    checks = "".join(f"<li class=\"{'pass' if ok else 'fail'}\">{html.escape(name)}: {'PASS' if ok else 'FAIL'}</li>" for name, ok in acceptance["checks"].items())
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{config['camera_id']} 机标准化验证</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei";margin:0;background:#f3f5f7;color:#17202a}}main{{max-width:1500px;margin:auto;padding:24px}}.hero,.card{{background:#fff;border-radius:14px;padding:20px;margin-bottom:20px;box-shadow:0 3px 16px #0001}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{width:100%;border-radius:8px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #dce2e7;padding:8px;text-align:left}}th{{background:#f7f9fa}}.pass{{color:#147d3f}}.warn{{color:#a15c00}}.reject,.fail{{color:#b42318}}code{{background:#eef1f3;padding:2px 5px}}@media(max-width:800px){{.images{{grid-template-columns:1fr}}}}</style></head>
    <body><main><section class="hero"><h1>{config['camera_id']} 机：18% 灰卡曝光 / 中性轴标准化</h1><p><strong>验收：{'PASS' if acceptance['passed'] else 'FAIL'}</strong>；profile 状态：<code>{config['color_profile']['status']}</code>。母版为不裁切 float32 线性 sRGB；预览曲线不写回母版。</p><ul>{checks}</ul><p>当前数据只支持曝光与中性轴可比，不宣称彩色区域绝对色度或 A/B 严格颜色一致。</p><img src="report_assets/quality_trends.png"></section>{''.join(cards)}
    <section class="hero"><h2>文件</h2><p><a href="correction_log.csv">校正日志</a> · <a href="color_metrics.csv">颜色提取</a> · <a href="run_manifest.json">运行清单</a> · <a href="acceptance.json">验收 JSON</a></p></section></main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def build_dual_report(camera_results: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for _, camera_rows, _ in camera_results for row in camera_rows]
    _trend_plot(rows, output_dir / "dual_camera_trends.png", "A/B validation trends")
    with (output_dir / "combined_correction_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    overall = all(acc["passed"] for _, _, acc in camera_results)
    scope_label = "全量门控" if any(cfg.get("run_mode") == "full" for cfg, _, _ in camera_results) else "十帧门控"
    summaries = "".join(f"<tr><td>{cfg['camera_id']}</td><td>{'PASS' if acc['passed'] else 'FAIL'}</td><td>{sum(r['qc_status']=='PASS' for r in camera_rows)} PASS / {sum(r['qc_status']=='WARN' for r in camera_rows)} WARN / {sum(r['qc_status']=='REJECT' for r in camera_rows)} REJECT</td><td><a href=\"{Path(cfg['output_dir']).name}/report.html\">详细报告</a></td></tr>" for cfg, camera_rows, acc in camera_results)
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>双相机标准化总报告</title><style>body{{font-family:system-ui,"Microsoft YaHei";max-width:1200px;margin:auto;padding:28px;background:#f3f5f7}}section{{background:white;padding:22px;border-radius:14px;margin-bottom:20px}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:9px}}</style></head><body><section><h1>双相机曝光、白平衡与中性轴标准化</h1><p><strong>{scope_label}：{'PASS' if overall else 'FAIL'}</strong></p><p>18% 灰卡定义分析母版；90% 白卡仅作诊断和预览锚点。两台相机仍为 <code>neutral_only</code>，没有 ColorChecker profile，不可宣称彩色区域严格跨相机一致。</p><table><tr><th>相机</th><th>验收</th><th>逐帧 QC</th><th>详情</th></tr>{summaries}</table></section><section><img src="dual_camera_trends.png"></section><section><a href="combined_correction_log.csv">combined_correction_log.csv</a></section></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def build_batch_report(config: dict[str, Any], rows: list[dict[str, Any]], color_rows: list[dict[str, Any]], acceptance: dict[str, Any], output_dir: Path) -> None:
    del color_rows
    trend = output_dir / "batch_trends.png"
    _trend_plot(rows, trend, f"Camera {config['camera_id']} full-series trends")
    counts = {status: sum(r["qc_status"] == status for r in rows) for status in ("PASS", "WARN", "REJECT")}
    checks = "".join(f"<li>{html.escape(name)}: {'PASS' if ok else 'FAIL'}</li>" for name, ok in acceptance["checks"].items())
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{config['camera_id']} 机全量报告</title><style>body{{font-family:system-ui,"Microsoft YaHei";max-width:1200px;margin:auto;padding:28px;background:#f3f5f7}}section{{background:#fff;padding:22px;border-radius:14px;margin-bottom:20px}}img{{width:100%}}code{{background:#eef1f3;padding:2px 5px}}</style></head><body><section><h1>{config['camera_id']} 机全量标准化（{len(rows)} 张）</h1><p>门控：<strong>{'PASS' if acceptance['passed'] else 'FAIL'}</strong>；逐帧 {counts['PASS']} PASS / {counts['WARN']} WARN / {counts['REJECT']} REJECT。</p><p>状态 <code>{config['color_profile']['status']}</code>：仅保证曝光和中性轴可比；预览不用于定量。</p><ul>{checks}</ul></section><section><img src="batch_trends.png"></section><section><a href="correction_log.csv">校正日志</a> · <a href="color_metrics.csv">颜色提取</a> · <a href="run_manifest.json">运行清单</a> · <a href="acceptance.json">验收 JSON</a></section></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
