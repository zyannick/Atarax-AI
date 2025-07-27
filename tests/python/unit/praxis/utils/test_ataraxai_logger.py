import logging
import os
import pytest
from unittest import mock
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger, CustomFormatter

@pytest.fixture
def logger(tmp_path):
    log_file = tmp_path / "test.log"
    return AtaraxAILogger(str(log_file)).get_logger(), log_file

def test_logger_writes_to_file(logger):
    log, log_file = logger
    log.info("test info message")
    log.error("test error message")
    with open(log_file, "r") as f:
        content = f.read()
    assert "test info message" in content
    assert "test error message" in content

def test_logger_levels(logger):
    log, log_file = logger
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")
    log.error("error message")
    log.critical("critical message")
    with open(log_file, "r") as f:
        content = f.read()
    assert "debug message" in content
    assert "info message" in content
    assert "warning message" in content
    assert "error message" in content
    assert "critical message" in content


def test_custom_formatter_formats_levels():
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=10,
        msg="info message", args=(), exc_info=None
    )
    formatter = CustomFormatter()
    formatted = formatter.format(record)
    assert "info message" in formatted
    assert "\x1b[34;20m" in formatted  # blue for INFO

def test_custom_formatter_all_levels():
    formatter = CustomFormatter()
    for level, color in [
        (logging.DEBUG, CustomFormatter.dard_blue),
        (logging.INFO, CustomFormatter.blue),
        (logging.WARNING, CustomFormatter.brown),
        (logging.ERROR, CustomFormatter.light_red),
        (logging.CRITICAL, CustomFormatter.bold_red),
    ]:
        record = logging.LogRecord(
            name="test", level=level, pathname=__file__, lineno=1,
            msg="msg", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert color in formatted
        assert "msg" in formatted

