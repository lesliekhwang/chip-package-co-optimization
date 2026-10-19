"""Central logger script and formmating.

In main run "log_init()", in subsequent scripts run "logger=log_get(__name__)".
"""
import logging

def log_init(level=logging.INFO):
    """Configures root logger for entire application."""
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] [%(asctime)s] [%(name)s:%(lineno)d] "%(message)s"',
        datefmt='%d-%m-%Y %H:%M:%S'
    )

def log_get(module_name: str) -> logging.Logger:
    """Returns a named logger instance for the calling module."""
    return logging.getLogger(module_name)