from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from backend.app.schemas.dataset import DatasetManifest, DatasetManifestRow
from backend.app.services.audio_validation import (
    DatasetDurationOutOfRange,
    LocalAudioPathError,
    resolve_local_audio_path,
    validate_training_duration,
)


class AuthorizedDatasetError(ValueError):
    """Raised when the explicit local authorized dataset contract is invalid."""


def load_authorized_manifest(
    manifest_path: Path | str,
    storage_root: Path | str = Path("data"),
) -> DatasetManifest:
    """Load a JSON/JSONL manifest and verify every referenced audio stays local."""
    root = Path(storage_root).expanduser().resolve()
    path = Path(manifest_path).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AuthorizedDatasetError("authorized manifest must stay under local storage") from exc
    if not path.is_file():
        raise AuthorizedDatasetError("authorized manifest file does not exist")

    try:
        rows, dataset_id = _read_rows(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AuthorizedDatasetError("authorized manifest is not valid JSON or JSONL") from exc
    if not rows:
        raise AuthorizedDatasetError("authorized manifest contains no rows")
    if len({row.segment_id for row in rows}) != len(rows):
        raise AuthorizedDatasetError("authorized manifest contains duplicate segment ids")
    for row in rows:
        if not row.text.strip():
            raise AuthorizedDatasetError("authorized manifest contains a row without transcript text")
        try:
            resolve_local_audio_path(row.path, root)
        except LocalAudioPathError as exc:
            raise AuthorizedDatasetError(
                f"authorized audio row is outside local storage or missing: {row.path}"
            ) from exc
    effective_seconds = round(sum(row.duration_seconds for row in rows), 6)
    try:
        validate_training_duration(effective_seconds)
    except DatasetDurationOutOfRange as exc:
        raise AuthorizedDatasetError(str(exc)) from exc
    return DatasetManifest(
        dataset_id=dataset_id,
        effective_seconds=effective_seconds,
        rows=rows,
        manifest_path=path,
    )


def _read_rows(path: Path) -> tuple[list[DatasetManifestRow], str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("authorized manifest JSON must be an object")
        raw_rows = payload.get("rows")
        raw_dataset_id = payload.get("dataset_id")
        if not isinstance(raw_rows, list) or not isinstance(raw_dataset_id, str):
            raise ValueError("authorized manifest JSON must contain dataset_id and rows")
        return [DatasetManifestRow.model_validate(row) for row in raw_rows], raw_dataset_id.strip()

    rows: list[DatasetManifestRow] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(DatasetManifestRow.model_validate(json.loads(line)))
    dataset_id = path.parent.name.strip()
    if not dataset_id:
        raise ValueError("authorized JSONL manifest has no dataset id")
    return rows, dataset_id
