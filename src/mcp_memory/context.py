from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ApplicationContext:
    memory_manager: Any = None
    memory_search: Any = None