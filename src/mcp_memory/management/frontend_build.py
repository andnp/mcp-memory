from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


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

    if not _dashboard_build_is_stale(frontend_root=frontend_root, dist_index_path=dist_index_path):
        return DashboardFrontendBuildResult(
            status="up_to_date",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
        )

    command = ["npm", "run", "build"]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=frontend_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return DashboardFrontendBuildResult(
            status="build_unavailable",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            message=str(exc),
        )

    stdout_tail = _tail_text(completed.stdout)
    stderr_tail = _tail_text(completed.stderr)
    message_parts = [part for part in (stdout_tail, stderr_tail) if part]
    if completed.returncode != 0:
        return DashboardFrontendBuildResult(
            status="build_failed",
            frontend_root=frontend_root,
            dist_index_path=dist_index_path,
            returncode=completed.returncode,
            message="\n".join(message_parts) or None,
        )

    return DashboardFrontendBuildResult(
        status="built",
        frontend_root=frontend_root,
        dist_index_path=dist_index_path,
        built=True,
        returncode=completed.returncode,
        message="\n".join(message_parts) or None,
    )


def _dashboard_build_is_stale(*, frontend_root: Path, dist_index_path: Path) -> bool:
    if not dist_index_path.exists():
        return True
    latest_source_mtime = _latest_dashboard_source_mtime(frontend_root)
    return latest_source_mtime > dist_index_path.stat().st_mtime


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