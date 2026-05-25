"""
BarkSSL Logger Module
Unified logging with file and console output.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """
    Unified logger for BarkSSL with console and file output.

    Usage:
        logger = Logger('train', 'logs')
        logger.info("Training started")
        logger.info(f"Loss: {loss:.4f}")
    """

    _instances = {}

    def __init__(
        self,
        name: str = 'barkssl',
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        console: bool = True,
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            log_dir: Directory for log files (None = no file output)
            level: Logging level
            console: Whether to output to console
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'{name}_{timestamp}.log'

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.log_file = log_file
        else:
            self.log_file = None

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.logger.critical(msg)

    def __call__(self, msg: str, level: str = 'info') -> None:
        """
        Log message with specified level.

        Args:
            msg: Message to log
            level: Log level (debug, info, warning, error, critical)
        """
        getattr(self.logger, level)(msg)

    @classmethod
    def get(cls, name: str = 'barkssl', log_dir: Optional[str] = None) -> 'Logger':
        """
        Get or create logger instance.

        Args:
            name: Logger name
            log_dir: Log directory

        Returns:
            Logger instance
        """
        key = f"{name}_{log_dir}"
        if key not in cls._instances:
            cls._instances[key] = cls(name, log_dir)
        return cls._instances[key]


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.

    Usage:
        tb_logger = TensorBoardLogger('logs/experiment')
        tb_logger.log_scalar('loss/train', loss, step)
        tb_logger.log_scalar('accuracy/val', acc, step)
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self._writer = None

    @property
    def writer(self):
        """Lazy load TensorBoard writer."""
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(self.log_dir)
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                return None
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: int) -> None:
        """Log multiple scalar values."""
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_audio(self, tag: str, audio: list, sample_rate: int, step: int) -> None:
        """Log audio."""
        if self.writer:
            self.writer.add_audio(tag, audio, step, sample_rate)

    def log_image(self, tag: str, image, step: int) -> None:
        """Log image."""
        if self.writer:
            self.writer.add_image(tag, image, step)

    def close(self) -> None:
        """Close the writer."""
        if self._writer:
            self._writer.close()


def setup_logger(
    name: str = 'barkssl',
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> Logger:
    """
    Setup and return a logger.

    Args:
        name: Logger name
        log_dir: Log directory
        level: Logging level

    Returns:
        Logger instance
    """
    return Logger.get(name, log_dir)


# Convenience function for quick logging
def get_logger(name: str = 'barkssl', log_dir: Optional[str] = None) -> Logger:
    """Get logger instance."""
    return Logger.get(name, log_dir)