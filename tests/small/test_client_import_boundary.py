from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.small


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_proxy_entrypoint_does_not_import_full_cli() -> None:
    result = _run_python(
        """
import sys
import mcp_memory.entrypoint as entrypoint

sys.argv = ["mcp-memory", "run", "--help"]
try:
    entrypoint.main()
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
print(f"full_cli_loaded={'mcp_memory.cli' in sys.modules}")
"""
    )

    assert result.returncode == 0, result.stderr
    assert "full_cli_loaded=False" in result.stdout


def test_server_import_does_not_load_daemon_or_model_modules() -> None:
    result = _run_python(
        """
import sys
import mcp_memory.server

for module_name in (
    "mcp_memory.daemon",
    "mcp_memory.daemon_app",
    "mcp_memory.daemon_transport",
    "mcp_memory.mcp.runtime",
    "sentence_transformers",
    "torch",
    "transformers",
    "ollama",
):
    if module_name in sys.modules:
        raise SystemExit(f"unexpected import: {module_name}")
print("proxy_imports_clean=True")
"""
    )

    assert result.returncode == 0, result.stderr
    assert "proxy_imports_clean=True" in result.stdout
