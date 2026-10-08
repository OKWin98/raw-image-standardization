from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def enforce_output_separation(source_root: Path, *write_roots: Path) -> None:
    source = source_root.resolve()
    for root in write_roots:
        resolved = root.resolve()
        if resolved == source or is_within(resolved, source):
            raise RuntimeError(f"Refusing to write inside source directory: {resolved}")


def require_absolute_directory(path: Path, label: str, *, must_exist: bool) -> Path:
    """Validate and normalize a user-supplied directory without creating it."""
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    resolved = path.resolve(strict=must_exist)
    if must_exist and not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    if resolved.parent == resolved:
        raise ValueError(f"{label} cannot be a drive root: {resolved}")
    return resolved


def enforce_disjoint_output(output_root: Path, source_root: Path, repository_root: Path) -> None:
    """Require the run directory, source tree, and repository to be disjoint."""
    output = output_root.resolve(strict=False)
    source = source_root.resolve()
    repository = repository_root.resolve()
    for protected, label in ((source, "source directory"), (repository, "repository")):
        if output == protected or is_within(output, protected) or is_within(protected, output):
            raise RuntimeError(f"Output directory must be separate from the {label}: {output}")


def prepare_output_directory(output_root: Path, *, resume: bool, dry_run: bool) -> Path:
    """Validate output-directory state and create it only for a real run."""
    output = output_root.resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError(
            f"Output directory is not empty; choose a new directory or pass --resume: {output}"
        )
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
    return output


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def snapshots(paths: Iterable[Path]) -> list[dict[str, str | int]]:
    return [file_snapshot(path) for path in paths]


def verify_snapshots(before: list[dict[str, str | int]]) -> None:
    for expected in before:
        current = file_snapshot(Path(str(expected["path"])))
        for field in ("size", "mtime_ns", "sha256"):
            if current[field] != expected[field]:
                raise RuntimeError(
                    f"Source file changed during processing: {expected['path']} ({field})"
                )


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise RuntimeError(f"Existing sample copy differs; refusing overwrite: {destination}")
        return
    shutil.copy2(source, destination)
    if sha256_file(destination) != source_hash:
        raise RuntimeError(f"Copy verification failed: {destination}")


def write_snapshot(path: Path, data: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

