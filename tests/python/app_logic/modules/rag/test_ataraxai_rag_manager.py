import pytest
from unittest import mock
from pathlib import Path
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager



@pytest.fixture
def mock_rag_config_manager():
    m = mock.Mock()
    m.get.side_effect = lambda key, default=None: {
        "rag_embedder_model": "test-embedder",
        "rag_use_reranking": False,
        "n_result": 5,
        "n_result_final": 3,
        "use_hyde": True,
        "rag_chunk_config": {},
        "rag_cross_encoder_model": "test-cross-encoder",
    }.get(key, default)
    return m

@pytest.fixture
def mock_core_ai_service():
    m = mock.Mock()
    m.generate_completion.return_value = "Hypothetical answer."
    return m

@pytest.fixture
def tmp_app_data_root(tmp_path):
    return tmp_path

@pytest.fixture
def manager(mock_rag_config_manager, tmp_app_data_root, mock_core_ai_service):
    with mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.AtaraxAIEmbedder"), \
         mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.RAGStore") as rag_store_cls, \
         mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.RAGManifest"):
        rag_store = rag_store_cls.return_value
        rag_store.collection_name = "ataraxai_knowledge"
        rag_store.client.get_or_create_collection.return_value = mock.Mock()
        rag_store.client.delete_collection.return_value = None
        rag_store.query.return_value = {"documents": [["doc1", "doc2", "doc3"]]}
        return AtaraxAIRAGManager(
            rag_config_manager=mock_rag_config_manager,
            app_data_root_path=tmp_app_data_root,
            core_ai_service=mock_core_ai_service,
        )

def test_check_manifest_validity(manager):
    manager.manifest.is_valid.return_value = True
    assert manager.check_manifest_validity() is True
    manager.manifest.is_valid.return_value = False
    assert manager.check_manifest_validity() is False

def test_rebuild_index_success(manager):
    manager.perform_initial_scan = mock.Mock(return_value=2)
    result = manager.rebuild_index(["/tmp"])
    assert result is True
    manager.perform_initial_scan.assert_called_once()

def test_rebuild_index_no_dirs(manager):
    result = manager.rebuild_index([])
    assert result is False

def test_perform_initial_scan_no_dirs(manager):
    assert manager.perform_initial_scan([]) == 0

def test_perform_initial_scan_files(tmp_path, manager):
    d = tmp_path / "dir"
    d.mkdir()
    f = d / "file.txt"
    f.write_text("test")
    manager.manifest.is_file_in_manifest.return_value = False
    files_found = manager.perform_initial_scan([str(d)])
    assert files_found == 1

def test_start_file_monitoring_success(manager):
    with mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.start_rag_file_monitoring") as start_monitor:
        start_monitor.return_value = mock.Mock(is_alive=lambda: True)
        result = manager.start_file_monitoring(["/tmp"])
        assert result is True

def test_start_file_monitoring_no_dirs(manager):
    result = manager.start_file_monitoring([])
    assert result is False

def test_stop_file_monitoring(manager):
    observer = mock.Mock(is_alive=lambda: True)
    manager.file_observer = observer
    manager.stop_file_monitoring()
    observer.stop.assert_called_once()
    observer.join.assert_called_once()

def test_stop_file_monitoring_no_active(manager):
    manager.file_observer = None
    # Should not raise
    manager.stop_file_monitoring()

def test_cleanup(manager):
    manager._cross_encoder = mock.Mock()
    manager._cross_encoder_initialized = True
    manager.stop_file_monitoring = mock.Mock()
    manager.cleanup()
    assert manager._cross_encoder is None
    assert manager._cross_encoder_initialized is False
    manager.stop_file_monitoring.assert_called_once()

def test_cross_encoder_lazy_load(manager):
    manager.rag_use_reranking = True
    manager.core_ai_service = True
    with mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.CrossEncoder") as ce:
        ce.return_value = "cross_encoder_instance"
        manager._cross_encoder_initialized = False
        result = manager.cross_encoder
        assert result == "cross_encoder_instance"

def test_generate_hypothetical_document(manager):
    out = manager._generate_hypothetical_document("What is AI?")
    assert out == "Hypothetical answer."

def test_rerank_documents_no_cross_encoder(manager):
    manager._cross_encoder = None
    manager._cross_encoder_initialized = True
    docs = ["a", "b"]
    out = manager._rerank_documents("q", docs)
    assert out == docs

def test_rerank_documents_with_cross_encoder(manager):
    ce = mock.Mock()
    ce.predict.return_value = [0.2, 0.9]
    manager._cross_encoder = ce
    manager._cross_encoder_initialized = True
    docs = ["a", "b"]
    out = manager._rerank_documents("q", docs)
    assert out == ["b", "a"]

def test_query_knowledge_simple(manager):
    manager._should_use_advanced_retrieval = mock.Mock(return_value=False)
    manager._simple_query = mock.Mock(return_value=["doc1"])
    result = manager.query_knowledge("test")
    assert result == ["doc1"]

def test_query_knowledge_advanced(manager):
    manager._should_use_advanced_retrieval = mock.Mock(return_value=True)
    manager._advanced_query = mock.Mock(return_value=["doc2"])
    result = manager.query_knowledge("test")
    assert result == ["doc2"]

def test_query_knowledge_empty_query(manager):
    with pytest.raises(ValueError):
        manager.query_knowledge("")

def test_simple_query_returns_docs(manager):
    manager.rag_store.query.return_value = {"documents": [["doc1", "doc2"]]}
    docs = manager._simple_query("q", None)
    assert docs == ["doc1", "doc2"]

def test_simple_query_returns_empty(manager):
    manager.rag_store.query.return_value = {"documents": []}
    docs = manager._simple_query("q", None)
    assert docs == []

def test_advanced_query_with_hyde_and_reranking(manager):
    manager.use_hyde = True
    manager.rag_use_reranking = True
    manager._generate_hypothetical_document = mock.Mock(return_value="hyde doc")
    manager.rag_store.query.return_value = {"documents": [["d1", "d2", "d3"]]}
    manager._rerank_documents = mock.Mock(return_value=["d3", "d2", "d1"])
    manager.n_result_final = 2
    docs = manager._advanced_query("q", None)
    assert docs == ["d3", "d2"]

def test_get_stats(manager):
    manager.rag_store.collection.count.return_value = 10
    manager.manifest.get_all_files.return_value = ["a", "b"]
    manager.file_observer = mock.Mock(is_alive=lambda: True)
    manager.worker_thread = mock.Mock(is_alive=lambda: True)
    stats = manager.get_stats()
    assert stats["total_documents"] == 10
    assert stats["manifest_files"] == 2
    assert stats["monitoring_active"] is True
    assert stats["worker_thread_active"] is True
