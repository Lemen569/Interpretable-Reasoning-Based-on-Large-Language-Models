import logging
import os
from typing import Optional


def setup_logger(
        name: str = "TKL-XR",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
) -> logging.Logger:
    """
    Configure a project logger with console and optional file output
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log file path is provided)
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance for the project
default_logger = setup_logger(
    name="TKL-XR",
    log_level=logging.INFO,
    log_file="./logs/tkl_xr.log"
)

if __name__ == "__main__":
    # Test logger configuration
    test_logger = setup_logger(name="TestLogger", log_level=logging.DEBUG)
    test_logger.debug("Debug message test")
    test_logger.info("Info message test")
    test_logger.warning("Warning message test")
    test_logger.error("Error message test")
    print("Logger setup test passed successfully!")