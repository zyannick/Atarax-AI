import hashlib
from pathlib import Path
import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;5;8m"
    blue = "\x1b[34;20m"
    dard_blue = "\x1b[38;5;20m"
    yellow = "\x1b[33;20m"
    brown = "\x1b[38;5;94m"
    red = "\x1b[31;20m"
    maroon = "\x1b[38;5;52m"
    bold_red = "\x1b[31;1m"
    very_dark_grey = "\x1b[38;5;232m"
    medium_grey = "\x1b[38;5;244m"
    light_grey = "\x1b[38;5;252m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: dard_blue + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: brown + format + reset,
        logging.ERROR: maroon + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ArataxAILogger:

    def __init__(self, log_file: str = "ataraxai.log"):
        self.logger = logging.getLogger("AtaraxaiLogger")
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter())

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)
