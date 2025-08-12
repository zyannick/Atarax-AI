import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.modules.prompt_engine.prompt_manager import PromptManager


@pytest.fixture
def mock_logger():
    return mock.Mock()


@pytest.fixture
def tmp_prompts_dir(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test_template.txt").write_text("Hello {name}!", encoding="utf-8")
    return prompts_dir


@pytest.fixture
def prompt_manager(tmp_prompts_dir, mock_logger):
    return PromptManager(prompts_directory=tmp_prompts_dir, logger=mock_logger)


def test_init_with_valid_directory(tmp_prompts_dir, mock_logger):
    pm = PromptManager(prompts_directory=tmp_prompts_dir, logger=mock_logger)
    assert pm.prompts_dir == tmp_prompts_dir


def test_init_with_invalid_directory(tmp_path):
    invalid_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        PromptManager(prompts_directory=invalid_dir)


def test_load_template_success(prompt_manager):
    result = prompt_manager.load_template("test_template", name="World")
    assert result == "Hello World!"


def test_load_template_caches(prompt_manager):
    with mock.patch("builtins.open", wraps=open) as m:
        prompt_manager.load_template("test_template", name="World")
        assert m.call_count == 1
        prompt_manager.load_template("test_template", name="World")
        assert m.call_count == 1  


def test_load_template_missing_file(prompt_manager):
    with pytest.raises(FileNotFoundError):
        prompt_manager.load_template("nonexistent_template")


def test_load_template_missing_placeholder(prompt_manager):
    result = prompt_manager.load_template("test_template")
    assert result == "Hello {name}!"
    prompt_manager.logger.warning.assert_called()


def test_clear_cache(prompt_manager):
    prompt_manager._cache["foo"] = "bar"
    prompt_manager.clear_cache()
    assert prompt_manager._cache == {}
    prompt_manager.logger.info.assert_called()


def test_get_cached_templates(prompt_manager):
    prompt_manager._cache = {"a": "1", "b": "2"}
    cached = prompt_manager.get_cached_templates()
    assert set(cached) == {"a", "b"}


def test_template_exists(prompt_manager):
    assert prompt_manager.template_exists("test_template")
    assert not prompt_manager.template_exists("nonexistent_template")


def test_list_available_templates(prompt_manager, tmp_prompts_dir):
    (tmp_prompts_dir / "another.txt").write_text("Test", encoding="utf-8")
    templates = prompt_manager.list_available_templates()
    assert set(templates) >= {"test_template", "another"}


@pytest.fixture
def mock_core_ai_service_manager():
    mock_mgr = mock.Mock()
    mock_mgr.tokenize.side_effect = lambda x: list(x.split())
    mock_mgr.decode.side_effect = lambda tokens: " ".join(tokens)
    mock_mgr.config_manager.llama_config_manager.get_generation_params.return_value = (
        mock.Mock(n_predict=5)
    )
    return mock_mgr


@pytest.fixture
def mock_rag_config():
    mock_cfg = mock.Mock()
    mock_cfg.context_allocation_ratio = 0.6
    return mock_cfg


def test_build_prompt_within_limit_basic(
    prompt_manager, mock_core_ai_service_manager, mock_rag_config
):
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    rag_context = "Some context here"
    user_query = "What is up?"
    prompt_template = "{history}\n{context}\n{query}"
    context_limit = 20

    result = prompt_manager.build_prompt_within_limit(
        history,
        rag_context,
        user_query,
        prompt_template,
        context_limit,
        mock_core_ai_service_manager,
        mock_rag_config,
    )
    assert user_query in result
    assert "Some context" in result or "No relevant documents" in result


def test_build_prompt_within_limit_no_budget(
    prompt_manager, mock_core_ai_service_manager, mock_rag_config
):
    prompt_template = "{history}\n{context}\n{query}"
    result = prompt_manager.build_prompt_within_limit(
        [],
        "context",
        "query",
        prompt_template,
        5,
        mock_core_ai_service_manager,
        mock_rag_config,
    )
    assert result == "\n\nquery"


def test_truncate_text_to_budget(prompt_manager, mock_core_ai_service_manager):
    text = "one two three four five"
    truncated = prompt_manager._truncate_text_to_budget(
        text, 3, mock_core_ai_service_manager
    )
    assert truncated == "one two three"


def test_truncate_text_to_budget_no_text(prompt_manager, mock_core_ai_service_manager):
    assert (
        prompt_manager._truncate_text_to_budget("", 3, mock_core_ai_service_manager)
        == ""
    )
    assert (
        prompt_manager._truncate_text_to_budget("text", 0, mock_core_ai_service_manager)
        == ""
    )


def test_build_history_within_budget_full(prompt_manager, mock_core_ai_service_manager):
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = prompt_manager._build_history_within_budget(
        history, 10, mock_core_ai_service_manager
    )
    assert "user:" in result and "assistant:" in result


def test_build_history_within_budget_invalid_message(
    prompt_manager, mock_core_ai_service_manager
):
    history = [{"foo": "bar"}]
    result = prompt_manager._build_history_within_budget(
        history, 10, mock_core_ai_service_manager
    )
    assert result == ""
    prompt_manager.logger.warning.assert_called()
