"""Small, dependency-free helpers for trustworthy pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MANIFEST_SCHEMA_VERSION = 1


class ProvenanceMismatchError(RuntimeError):
    """Raised when an existing artifact cannot be safely reused."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a large cube into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(
    path: str | Path,
    include_path: bool = True,
    include_mtime: bool = True,
) -> dict[str, Any]:
    """Describe a file strongly enough to detect stale or mixed pipeline inputs."""
    file_path = Path(path)
    stat = file_path.stat()
    identity: dict[str, Any] = {
        "name": file_path.name,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(file_path),
    }
    if include_mtime:
        identity["mtime_ns"] = stat.st_mtime_ns
    if include_path:
        identity["path"] = str(file_path.resolve())
    return identity


def manifest_path(artifact_path: str | Path) -> Path:
    """Return the JSON sidecar path for an artifact."""
    artifact = Path(artifact_path)
    return artifact.with_name(f"{artifact.name}.provenance.json")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a JSON file in the target directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def save_npy_atomic(path: str | Path, array: np.ndarray) -> Path:
    """Atomically save a NumPy array without exposing a partial .npy file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_artifact_manifest(
    artifact_path: str | Path,
    provenance: dict[str, Any],
    related_paths: Iterable[str | Path] = (),
) -> Path:
    """Record provenance plus hashes of the primary and companion artifacts."""
    artifact = Path(artifact_path)
    related = [Path(path) for path in related_paths]
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "artifact": file_identity(artifact, include_path=False),
        "related_artifacts": [file_identity(path, include_path=False) for path in related],
    }
    return write_json_atomic(manifest_path(artifact), payload)


def _load_manifest(artifact_path: Path) -> dict[str, Any]:
    sidecar = manifest_path(artifact_path)
    if not sidecar.is_file():
        raise ProvenanceMismatchError(
            f"Existing artifact has no provenance manifest: {artifact_path}. "
            "Regenerate it with overwrite=True/--overwrite."
        )
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceMismatchError(
            f"Cannot read provenance manifest {sidecar}: {exc}. Regenerate with overwrite enabled."
        ) from exc
    return payload


def _verify_identity(path: Path, recorded: dict[str, Any]) -> None:
    if not path.is_file():
        raise ProvenanceMismatchError(f"Recorded artifact is missing: {path}")
    actual = file_identity(path, include_path=False)
    stable_keys = ("name", "size_bytes", "sha256")
    if any(actual.get(key) != recorded.get(key) for key in stable_keys):
        raise ProvenanceMismatchError(
            f"Artifact content no longer matches its provenance manifest: {path}. "
            "Regenerate it with overwrite=True/--overwrite."
        )


def validate_reusable_artifact(
    artifact_path: str | Path,
    expected_provenance: dict[str, Any],
    related_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Refuse reuse unless provenance and all recorded file hashes still match."""
    artifact = Path(artifact_path)
    payload = _load_manifest(artifact)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ProvenanceMismatchError(
            f"Unsupported provenance schema for {artifact}: {payload.get('schema_version')!r}. "
            "Regenerate it with overwrite enabled."
        )
    if payload.get("provenance") != expected_provenance:
        raise ProvenanceMismatchError(
            f"Existing artifact was produced from different source data or settings: {artifact}. "
            "Regenerate it with overwrite=True/--overwrite."
        )

    _verify_identity(artifact, payload.get("artifact", {}))
    related_root = Path(related_directory) if related_directory is not None else artifact.parent
    for identity in payload.get("related_artifacts", []):
        _verify_identity(related_root / identity["name"], identity)
    return payload
