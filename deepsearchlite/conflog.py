import logging
from pathlib import Path

from concurrent_log_handler import ConcurrentRotatingFileHandler


def setup_logger(log_file: Path) -> logging.Logger:

    log_level = logging.DEBUG

    # Create a logger
    logger = logging.getLogger("deep-search-lite")
    logger.setLevel(log_level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Create a file handler
    file_handler = ConcurrentRotatingFileHandler(str(log_file), "a", 1024 * 1024, 5)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
