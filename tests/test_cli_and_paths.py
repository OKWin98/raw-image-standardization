import json
from pathlib import Path

import pytest

from imagecal.__main__ import build_parser
from imagecal.pipeline import prepare_run_root
from imagecal.safety import (
    enforce_disjoint_output,
    prepare_output_directory,
    require_absolute_directory,
)


def test_raw_commands_require_source_and_output_paths():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run-full", "--config", "config.yaml"])


def test_relative_runtime_path_is_rejected():
    with pytest.raises(ValueError, match="absolute"):
        require_absolute_directory(Path("relative/path"), "--output-dir", must_exist=False)


def test_output_must_be_disjoint_from_source_and_repository(tmp_path: Path):
    source = tmp_path / "原始 照片"
    repository = tmp_path / "代码 库"
    source.mkdir()
    repository.mkdir()
    with pytest.raises(RuntimeError, match="source"):
        enforce_disjoint_output(source / "run", source, repository)
    with pytest.raises(RuntimeError, match="repository"):
        enforce_disjoint_output(repository / "run", source, repository)


def test_nonempty_output_requires_resume(tmp_path: Path):
    output = tmp_path / "运行 输出"
    output.mkdir()
    (output / "existing.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not empty"):
        prepare_output_directory(output, resume=False, dry_run=False)
    assert prepare_output_directory(output, resume=True, dry_run=False) == output.resolve()


def test_dry_run_does_not_create_output(tmp_path: Path):
    output = tmp_path / "尚未创建 输出"
    prepare_output_directory(output, resume=False, dry_run=True)
    assert not output.exists()


def test_resume_requires_matching_manifest(tmp_path: Path):
    source = tmp_path / "source"
    repository = tmp_path / "repository"
    output = tmp_path / "output"
    source.mkdir()
    repository.mkdir()
    output.mkdir()
    config = tmp_path / "camera.yaml"
    config.write_text("camera_id: A\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "imagecal_version": "wrong-version",
                "command": "run",
                "source_root": str(source.resolve()),
                "configs": [],
            }
        ),
        encoding="utf-8",
    )
    import imagecal.pipeline as pipeline

    original = pipeline.REPOSITORY_ROOT
    pipeline.REPOSITORY_ROOT = repository
    try:
        with pytest.raises(RuntimeError, match="manifest mismatch"):
            prepare_run_root(
                command="run",
                config_paths=[config],
                source_root=source,
                output_root=output,
                resume=True,
                dry_run=False,
            )
    finally:
        pipeline.REPOSITORY_ROOT = original
