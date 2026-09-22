import os
from pathlib import Path

from backend.app.core.errors import PathOutsideStorage


class LocalStorage:
    """Resolve and publish files under a local, area-scoped storage root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, area: str, relative: str | Path) -> Path:
        area_root = (self.root / area).resolve()
        candidate = (area_root / Path(relative)).resolve()
        try:
            candidate.relative_to(area_root)
        except ValueError as exc:
            raise PathOutsideStorage(
                f"Path is outside storage area '{area}'"
            ) from exc
        return candidate

    def atomic_publish(
        self, source: Path | str, area: str, relative: str | Path
    ) -> Path:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = self.resolve(area, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_path, destination)
        return destination
