"""Run the versioned local search-quality benchmark."""

from __future__ import annotations

import json

from search_quality.runner import run_in_process


def main() -> None:
    """Print deterministic search-quality metrics as JSON."""
    print(json.dumps(run_in_process().to_mapping(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
