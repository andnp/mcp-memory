"""Application-owned maintenance read port."""

from mcp_memory.core.ports.memory import MemoryMaintenanceReadPort

MaintenanceReadRepositoryLike = MemoryMaintenanceReadPort

__all__ = ["MaintenanceReadRepositoryLike"]
