from click.testing import CliRunner

from mcp_memory.cli import main


def test_dashboard_command_wires_management_app(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_create_management_app(project_override=None):
        captured["project_override"] = project_override
        return object()

    def fake_run(app, host: str, port: int, log_level: str) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("mcp_memory.cli.create_management_app", fake_create_management_app)
    monkeypatch.setattr("mcp_memory.cli.uvicorn.run", fake_run)

    result = runner.invoke(main, ["dashboard", "--host", "127.0.0.1", "--port", "9999", "--project", "demo"])

    assert result.exit_code == 0
    assert captured["project_override"] == "demo"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9999
    assert captured["log_level"] == "info"
