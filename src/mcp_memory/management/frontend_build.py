from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_BUILD_TIMEOUT_SECONDS = 120.0
_FAILURE_MARKER_NAME = ".build-failed-marker"


@dataclass(frozen=True)
class DashboardFrontendBuildResult:
    status: str
    frontend_root: Path
    dist_index_path: Path
    built: bool = False
    returncode: int | None = None
    message: str | None = None


def ensure_dashboard_frontend_built(*, static_root: Path) -> DashboardFrontendBuildResult:
    frontend_root = static_root.with_name("frontend")
    dist_index_path = static_root / "dist" / "index.html"
    if not frontend_root.exists():
        return DashboardFrontendBuildResult(
            status="frontend_root_missing",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            message="dashboard frontend root does not exist",
        )

    latest_source_mtime = _latest_dashboard_source_mtime(frontend_root)
    failure_marker_path = static_root / _FAILURE_MARKER_NAME
    if not _dashboard_build_is_stale(dist_index_path=dist_index_path, latest_source_mtime=latest_source_mtime):
        return DashboardFrontendBuildResult(
            status="up_to_date",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
        )

    if _build_previously_failed_for(failure_marker_path, latest_source_mtime):
        return DashboardFrontendBuildResult(
            status="build_failed_cached",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            message="skipping rebuild: previous attempt failed and sources are unchanged",
        )

    command = ["npm", "run", "build"]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=frontend_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        _record_build_failure(failure_marker_path, latest_source_mtime)
        return DashboardFrontendBuildResult(
            status="build_unavailable",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            message=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        _record_build_failure(failure_marker_path, latest_source_mtime)
        return DashboardFrontendBuildResult(
            status="build_timed_out",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            message=str(exc),
        )

    stdout_tail = _tail_text(completed.stdout)
    stderr_tail = _tail_text(completed.stderr)
    message_parts = [part for part in (stdout_tail, stderr_tail) if part]
    if completed.returncode != 0:
        _record_build_failure(failure_marker_path, latest_source_mtime)
        return DashboardFrontendBuildResult(
            status="build_failed",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            returncode=completed.returncode,
            message="\n".join(message_parts) or None,
        )

    failure_marker_path.unlink(missing_ok=True)
    return DashboardFrontendBuildResult(
        status="built",
        frontend_root=frontend_root,
        dist_index_path=dist_index_path,
        built=True,
        returncode=completed.returncode,
        message="\n".join(message_parts) or None,
    )


def _dashboard_build_is_stale(*, dist_index_path: Path, latest_source_mtime: float) -> bool:
    if not dist_index_path.exists():
        return True
    return latest_source_mtime > dist_index_path.stat().st_mtime


def _build_previously_failed_for(failure_marker_path: Path, latest_source_mtime: float) -> bool:
    try:
        recorded_mtime = float(failure_marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return recorded_mtime >= latest_source_mtime


def _record_build_failure(failure_marker_path: Path, latest_source_mtime: float) -> None:
    failure_marker_path.parent.mkdir(parents=True, exist_ok=True)
    failure_marker_path.write_text(str(latest_source_mtime), encoding="utf-8")


def _latest_dashboard_source_mtime(frontend_root: Path) -> float:
    latest_mtime = 0.0
    candidate_paths = [
        frontend_root / "package.json",
        frontend_root / "package-lock.json",
        frontend_root / "tsconfig.json",
        frontend_root / "tsconfig.node.json",
        frontend_root / "vite.config.ts",
        frontend_root / "postcss.config.js",
        frontend_root / "tailwind.config.js",
        frontend_root / "index.html",
    ]
    candidate_paths.extend(path for path in (frontend_root / "src").rglob("*") if path.is_file())
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        latest_mtime = max(latest_mtime, candidate_path.stat().st_mtime)
    return latest_mtime


def _tail_text(value: str | None, *, max_lines: int = 20) -> str | None:
    if not value:
        return None
    lines = [line for line in value.strip().splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])