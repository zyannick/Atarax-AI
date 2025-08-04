import logging
import os
from pathlib import Path
import pytest
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger, CustomFormatter

@pytest.fixture
def temp_log_dir(tmp_path):
    return tmp_path / "logs"


def test_logger_returns_same_logger_instance(temp_log_dir):
    logger_instance = AtaraxAILogger(log_file="test2.log", log_dir=temp_log_dir)
    logger1 = logger_instance.get_logger()
    logger2 = logger_instance.get_logger()
    assert logger1 is logger2

def test_custom_formatter_formats_levels():
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=10,
        msg="error occurred",
        args=(),
        exc_info=None,
    )
    formatter = CustomFormatter()
    formatted = formatter.format(record)
    assert "error occurred" in formatted
    assert "\x1b[38;5;196m" in formatted  

def test_log_dir_created(tmp_path):
    log_dir = tmp_path / "new_logs"
    assert not log_dir.exists()
    AtaraxAILogger(log_file="foo.log", log_dir=log_dir)
    assert log_dir.exists()