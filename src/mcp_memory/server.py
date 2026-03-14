import asyncio
import logging
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_memory.mcp.runtime import create_runtime

# Set environment variables for performance and output stability
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self, project_override: str | None = None):
        self.project_override = project_override
        self.server = Server("mcp-memory")
        self.ctx = None  # Will hold the Memory Context (ApplicationContext refined for memory)

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            # This will return the list of tools defined in the original project's memory_tools.py
            # Adapt the get_memory_tools logic here.
            from mcp_memory.mcp.tools import get_memory_tools
            return get_memory_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            # This will delegate call handling to the original project's handlers.py
            # Adapt the handlers logic here.
            from mcp_memory.mcp.handlers import call_memory_tool
            return await call_memory_tool(self.ctx, name, arguments)

    async def run(self) -> None:
        """Runs the MCP server using stdio."""
        logger.info("Initializing MCP Memory Server...")
        if self.ctx is None:
            self.ctx = await asyncio.to_thread(
                create_runtime,
                self.project_override,
                None,
            )

        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        finally:
            if self.ctx is not None:
                self.ctx.close()
