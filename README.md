# 双相机 RAW 图像标准化

本项目使用认证 18% 灰卡逐帧统一 RAW 的曝光和中性轴，生成用于定量分析的 float32 线性 sRGB D65 母版，以及仅用于人工检查的 sRGB 预览。90% 白卡用于诊断和预览曲线，不强制改变分析母版。

仓库只保存代码、配置、测试和文档。RAW、验证样本副本、TIFF、预览、报告、CSV、checkpoint 和虚拟环境必须保存在仓库以外。

## 安装

建议使用 Python 3.14：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\run_test.ps1
```

安装后可以使用 `imagecal`，也可以使用 `.\.venv\Scripts\python.exe -m imagecal`。所有读取 RAW 的命令都必须显式指定绝对输入路径和绝对输出路径。

## 先做无写入预检

```powershell
imagecal run-full `
  --config ".\configs\dual_validation.yaml" `
  --source-root "X:\RawData\experiment" `
  --output-dir "Y:\ImageCalRuns\2026-10-09-full" `
  --dry-run
```

预检会验证路径隔离、输入文件、A/B 相机数量、首张 RAW 可读性、预计未压缩母版体积和输出盘可用空间；不会创建输出目录或任何图像。

## 双机验证与全量处理

A/B 各五张验证：

```powershell
imagecal run-dual `
  --config ".\configs\dual_validation.yaml" `
  --source-root "X:\RawData\experiment" `
  --output-dir "Y:\ImageCalRuns\2026-10-09-validation"
```

630 张全量处理：

```powershell
imagecal run-full `
  --config ".\configs\dual_validation.yaml" `
  --source-root "X:\RawData\experiment" `
  --output-dir "Y:\ImageCalRuns\2026-10-09-full"
```

也可以调用 `run_validation.ps1` 或 `run_full.ps1`；两个脚本都强制接收 `SourceRoot` 和 `OutputDir`。

## 输出安全和续跑

- 输入、输出必须是绝对路径；输出不能位于原始照片目录或代码仓库内部，也不能包含它们。
- 新输出目录可自动创建。非空输出目录默认拒绝，防止覆盖已有实验结果。
- 中断后仅可对原目录显式加入 `--resume`。程序会核对根 manifest 中的命令、版本、源目录和配置哈希；不一致时拒绝续跑。
- 处理前后核对所用原文件的 SHA-256、大小和修改时间；程序不在原始目录创建 sidecar、缓存或输出。

运行目录结构：

```text
<output-dir>/
├── run_manifest.json
├── report.html
├── combined_correction_log.csv
├── work/validation/camera_A|B/   # 验证样本副本
├── camera_A/
└── camera_B/
```

每台相机目录包含 `linear_float_tiff/`、`previews/`、`correction_log.csv`、`color_metrics.csv`、报告、输入快照和 checkpoints。定量提取只能读取 float32 母版；预览 PNG/JPG 永不作为定量输入。

## 配置

公共物理标准位于 `configs/standards.yaml`；A/B 的 ROI、文件匹配规则和验证样本位于各自的 validation 配置。配置不包含个人盘符、输入路径或输出路径。ROI 使用 `[left, top, right, bottom]` 归一化坐标。

## 科研解释边界

当前 profile 状态为 `neutral_only`：结果支持曝光和中性轴比较，但没有实际光照下的 ColorChecker profile，不能宣称彩色区域绝对色度准确或 A/B 严格跨相机一致。Hue 使用圆周统计并排除低饱和度像素。未来应在试验体同一平面、相同光照下为每台相机分别拍摄 ColorChecker 并验证色差后，再升级为完整颜色标定。
