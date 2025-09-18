import logging
from pathlib import Path
from typing import Optional

class CustomFormatter(logging.Formatter):
    # from https://stackoverflow.com/a/56944256
    blue = "\x1b[34;20m"
    dard_blue = "\x1b[38;5;20m"
    brown = "\x1b[38;5;94m"
    maroon = "\x1b[38;5;52m"
    bold_red = "\x1b[31;1m"
    light_red = "\x1b[38;5;196m"
    reset = "\x1b[0m"
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: dard_blue + log_format + reset,
        logging.INFO: blue + log_format + reset,
        logging.WARNING: brown + log_format + reset,
        logging.ERROR: light_red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class AtaraxAILogger:

    def __init__(self, log_file: str = "ataraxai.log", log_dir: Optional[Path] = None):
        """
        Initializes the AtaraxaiLogger instance.

        Sets up a logger with both file and console handlers:
        - The file handler logs all messages at DEBUG level and above to the specified log file.
        - The console handler logs messages at INFO level and above to the console, using a custom formatter.
        - Log messages are formatted with timestamp, logger name, log level, and message.

        Args:
            log_file (str): Path to the log file. Defaults to "ataraxai.log".
            log_dir (Path | None): Directory to store log files. If None, uses the current directory.
        """
        self.logger = logging.getLogger("AtaraxaiLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        if log_dir is None:
            log_dir = Path("logs")
            
        log_dir.mkdir(parents=True, exist_ok=True)


        if not self.logger.handlers:
            file_handler = logging.FileHandler(str(log_dir / log_file))
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(CustomFormatter())

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: The logger instance configured with file and console handlers.
        """
        return self.logger
