import pytest
from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig

def test_default_values():
    config = RAGConfig()
    assert config.config_version == 1.0
    assert config.rag_model_path == ""
    assert config.rag_chunk_size == 400
    assert config.rag_chunk_overlap == 50
    assert config.rag_separators is None
    assert config.rag_keep_separator is True
    assert config.rag_model_name_for_tiktoken == "gpt-3.5-turbo"

def test_custom_values():
    config = RAGConfig(
        config_version=2.0,
        rag_model_path="some/path",
        rag_chunk_size=512,
        rag_chunk_overlap=64,
        rag_separators=["\n", "."],
        rag_keep_separator=False,
        rag_model_name_for_tiktoken="custom-model"
    )
    assert config.config_version == 2.0
    assert config.rag_model_path == "some/path"
    assert config.rag_chunk_size == 512
    assert config.rag_chunk_overlap == 64
    assert config.rag_separators == ["\n", "."]
    assert config.rag_keep_separator is False
    assert config.rag_model_name_for_tiktoken == "custom-model"

@pytest.mark.parametrize(
    "rag_model_path,expected",
    [
        ("", False),
        ("model/path", True),
    ]
)
def test_is_setup_complete(rag_model_path, expected):
    config = RAGConfig(rag_model_path=rag_model_path)
    assert config.is_setup_complete() == expected