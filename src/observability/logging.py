"""Structured logging configuration."""

import logging
import sys
from typing import cast

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog JSON logging.

    Args:
        log_level: Python logging level name.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger.

    Args:
        name: Logger name.

    Returns:
        Bound structlog logger.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
