"""HTTP rollout server for BrowserGym WebArena sessions."""

from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    level_name = os.getenv("ROLLOUT_SERVER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(level)
    package_logger.propagate = False

    if not package_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
        package_logger.addHandler(handler)
    for handler in package_logger.handlers:
        handler.setLevel(level)


configure_logging()
