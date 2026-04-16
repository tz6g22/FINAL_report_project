"""Project-relative path helpers shared by analysis and evaluation code."""

from __future__ import annotations

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Return the repository root for this project."""

    return _PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path against the repository root without requiring it to exist."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (_PROJECT_ROOT / candidate).resolve(strict=False)


def format_project_path(path: str | Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute path."""

    resolved = resolve_project_path(path)
    try:
        return str(resolved.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(resolved)
