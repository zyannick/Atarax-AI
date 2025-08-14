from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig
from unittest import mock
import pytest



def test_default_values():
    config = RAGConfig()
    assert config.rag_model_path == ""
    assert config.rag_chunk_size == 400
    assert config.rag_chunk_overlap == 50
    assert config.rag_watched_directories == set()
    assert config.rag_time_out_update == 30.0
    assert config.rag_separators is None
    assert config.rag_keep_separator is True
    assert config.rag_model_name_for_tiktoken == "gpt-3.5-turbo"
    assert config.rag_embedder_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.rag_use_reranking is False
    assert config.rag_cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.rag_n_result == 5
    assert config.rag_n_result_final == 3
    assert config.rag_use_hyde is True
    assert config.cross_encoder_hits == 5
    assert config.context_allocation_ratio == 0.5

def test_custom_values():
    config = RAGConfig(
        rag_model_path="/tmp/model",
        rag_chunk_size=100,
        rag_chunk_overlap=10,
        rag_watched_directories={"dir1", "dir2"},
        rag_time_out_update=60.0,
        rag_separators=[".", "!", "?"],
        rag_keep_separator=False,
        rag_model_name_for_tiktoken="custom-model",
        rag_embedder_model="custom-embedder",
        rag_use_reranking=True,
        rag_cross_encoder_model="custom-cross-encoder",
        rag_n_result=10,
        rag_n_result_final=7,
        rag_use_hyde=False,
        cross_encoder_hits=8,
        context_allocation_ratio=0.8,
    )
    assert config.rag_model_path == "/tmp/model"
    assert config.rag_chunk_size == 100
    assert config.rag_chunk_overlap == 10
    assert config.rag_watched_directories == {"dir1", "dir2"}
    assert config.rag_time_out_update == 60.0
    assert config.rag_separators == [".", "!", "?"]
    assert config.rag_keep_separator is False
    assert config.rag_model_name_for_tiktoken == "custom-model"
    assert config.rag_embedder_model == "custom-embedder"
    assert config.rag_use_reranking is True
    assert config.rag_cross_encoder_model == "custom-cross-encoder"
    assert config.rag_n_result == 10
    assert config.rag_n_result_final == 7
    assert config.rag_use_hyde is False
    assert config.cross_encoder_hits == 8
    assert config.context_allocation_ratio == 0.8

def test_is_setup_complete_exists():
    config = RAGConfig(rag_model_path="/fake/path/model")
    with mock.patch("pathlib.Path.exists", return_value=True):
        assert config.is_setup_complete() is True

def test_is_setup_complete_not_exists():
    config = RAGConfig(rag_model_path="/fake/path/model")
    with mock.patch("pathlib.Path.exists", return_value=False):
        assert config.is_setup_complete() is False
