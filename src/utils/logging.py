"""
Shared logging configuration for CLI entry points.

Library modules (src/data, src/models, src/features, src/evaluation) must
not call logging.basicConfig() at import time — that mutates the root
logger as a side effect of `import`, which fights any handler configured
by a consuming process (e.g. uvicorn's own logging setup in Phase 4).
Only __main__ blocks should call configure_logging().
"""
import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for standalone CLI runs."""
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")
