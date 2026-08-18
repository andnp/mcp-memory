from __future__ import annotations

import sys

_PROXY_COMMANDS = frozenset({"run", "internal-run"})


def _top_level_command(arguments: list[str]) -> str | None:
    for argument in arguments:
        if argument == "--debug":
            continue
        if argument.startswith("-"):
            return None
        return argument
    return None


def main() -> None:
    if _top_level_command(sys.argv[1:]) in _PROXY_COMMANDS:
        from mcp_memory.cli_proxy import main as proxy_main

        proxy_main()
        return

    from mcp_memory.cli import main as cli_main

    cli_main()
