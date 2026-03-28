"""Structured logging configuration for hdf2zarr."""

import logging
import sys

import structlog


def configure_logging(
    verbosity: int = 0,
    quiet: bool = False,
    json_output: bool = False,
) -> None:
    """Configure structlog for the application.

    Logs are sent to stderr to avoid interfering with Rich UI on stdout.

    Parameters
    ----------
    verbosity : int
        Verbosity level. 0 = WARNING (default), 1 = INFO, 2+ = DEBUG.
    quiet : bool
        If ``True``, only ERROR and above are emitted.
    json_output : bool
        If ``True``, emit JSON-formatted log lines instead of
        human-readable output.
    """
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
