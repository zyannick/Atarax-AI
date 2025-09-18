from pathlib import Path
from unittest import mock

import pytest

from ataraxai.praxis.modules.prompt_engine.prompt_manager import PromptManager


@pytest.fixture
def mock_logger():
    return mock.Mock()


@pytest.fixture
def tmp_prompts_dir(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_file = prompts_dir / "sample_template.txt"
    template_file.write_text("History: {history}\nContext: {context}\nQuery: {query}")
    return prompts_dir


def test_init_with_valid_directory(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    assert pm.prompts_dir == tmp_prompts_dir
    assert isinstance(pm._cache, dict)
    mock_logger.info.assert_called_once()


def test_init_with_invalid_directory(tmp_path: Path):
    invalid_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        PromptManager(invalid_dir)


def test_load_template_success(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    content = pm.load_template("sample_template")
    assert "History:" in content
    assert "Context:" in content
    assert "Query:" in content
    assert "sample_template" in pm._cache


def test_load_template_not_found(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    with pytest.raises(FileNotFoundError):
        pm.load_template("nonexistent_template")


def test_clear_cache(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    pm._cache["sample"] = "test"
    pm.clear_cache()
    assert pm._cache == {}
    mock_logger.info.assert_called_with("Template cache cleared")


def test_get_cached_templates(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    pm._cache["sample1"] = "a"
    pm._cache["sample2"] = "b"
    cached = pm.get_cached_templates()
    assert set(cached) == {"sample1", "sample2"}


def test_template_exists(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    assert pm.template_exists("sample_template")
    assert not pm.template_exists("missing_template")


def test_list_available_templates(tmp_prompts_dir: Path, mock_logger: mock.Mock):
    (tmp_prompts_dir / "another.txt").write_text("test")
    pm = PromptManager(tmp_prompts_dir, logger=mock_logger)
    templates = pm.list_available_templates()
    assert set(templates) == {"sample_template", "another"}

