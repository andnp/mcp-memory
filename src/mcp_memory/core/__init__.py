"""Core memory modules.

Keep package import side effects minimal while the prototype is being
incrementally stabilized. Higher-level system wiring can be reintroduced once
the runtime imports are cleaned up.
"""

from mcp_memory.core.pipeline import MemoryPipeline

__all__ = ["MemoryPipeline"]
