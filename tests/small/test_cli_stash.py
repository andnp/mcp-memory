from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp_memory.cli import main
from mcp_memory.config import resolve_workspace_id
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.small


def test_stash_uses_git_root_for_workspace_id(monkeypatch, tmp_path: Path) -> None:
    """
    Verify stash resolves the workspace ID from the enclosing git root.

    This keeps CLI capture aligned with the repo-level workspace semantics.
    """
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "feature"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()
    monkeypatch.chdir(nested)

    result = runner.invoke(main, ["memory", "stash", "remember", "the", "git", "root"])

    assert result.exit_code == 0
    assert "Thought stashed successfully (ID:" in result.output

    runtime = create_runtime(cwd=nested)
    try:
        assert runtime.journal is not None
        pending = runtime.journal.get_pending(workspace_id=runtime.workspace_id)
        resolved_workspace_id = runtime.workspace_id
    finally:
        runtime.close()

    assert [entry.content for entry in pending] == ["remember the git root"]
    assert [entry.workspace_id for entry in pending] == [resolve_workspace_id(cwd=nested)]
    assert resolved_workspace_id == resolve_workspace_id(cwd=nested)


def test_stash_reads_multiline_content_from_stdin(monkeypatch, tmp_path: Path) -> None:
    """
    Verify stash accepts multiline input from stdin when no text argument is provided.

    This preserves the zero-friction stash flow for longer notes.
    """
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    result = runner.invoke(
        main,
        ["memory", "stash", "--workspace-root", str(workspace)],
        input="first line\nsecond line\n",
    )

    assert result.exit_code == 0
    assert "Thought stashed successfully (ID:" in result.output

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.journal is not None
        recent = runtime.journal.get_recent(limit=1)
        resolved_workspace_id = runtime.workspace_id
    finally:
        runtime.close()

    assert len(recent) == 1
    assert recent[0].content == "first line\nsecond line"
    assert recent[0].workspace_id == resolved_workspace_id


def test_stash_rejects_empty_input(monkeypatch, tmp_path: Path) -> None:
    """
    Verify stash fails fast when neither arguments nor stdin provide content.

    This keeps the CLI contract explicit instead of silently recording nothing.
    """
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    result = runner.invoke(main, ["memory", "stash", "--workspace-root", str(workspace)], input="   ")

    assert result.exit_code == 1
    assert "Provide stash text as arguments or via stdin." in result.output
