import pytest
from unittest import mock
from pathlib import Path
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager

@pytest.fixture
def mock_preferences_manager():
    mock_pm = mock.Mock()
    mock_pm.get.return_value = "sentence-transformers/all-MiniLM"
    return mock_pm

@pytest.fixture
def tmp_app_data_root(tmp_path):
    return tmp_path / "app_data"

@pytest.fixture
def manager(mock_preferences_manager, tmp_app_data_root):
    tmp_app_data_root.mkdir(parents=True, exist_ok=True)
    with mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.AtaraxAIEmbedder") as MockEmbedder, \
         mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.RAGStore") as MockRAGStore, \
         mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.RAGManifest") as MockManifest:
        MockEmbedder.return_value = mock.Mock()
        MockRAGStore.return_value = mock.Mock(collection=mock.Mock())
        MockManifest.return_value = mock.Mock()
        yield AtaraxAIRAGManager(mock_preferences_manager, tmp_app_data_root)

def test_init_creates_paths_and_instances(manager, tmp_app_data_root):
    rag_store_db_path = tmp_app_data_root / "rag_chroma_store"
    manifest_file_path = tmp_app_data_root / "rag_manifest.json"
    assert rag_store_db_path.exists()
    assert isinstance(manager.app_data_root_path, Path)
    assert manager.manifest_file_path == manifest_file_path

def test_start_file_monitoring_starts_observer(manager):
    with mock.patch("ataraxai.app_logic.modules.rag.ataraxai_rag_manager.start_rag_file_monitoring") as mock_start_monitor:
        mock_observer = mock.Mock()
        mock_start_monitor.return_value = mock_observer
        manager.start_file_monitoring(["/some/dir"])
        assert manager.file_observer == mock_observer
        mock_start_monitor.assert_called_once()

def test_start_file_monitoring_no_directories(manager, capsys):
    manager.start_file_monitoring([])
    captured = capsys.readouterr()
    assert "No directories specified to watch for RAG updates." in captured.out

def test_stop_file_monitoring_when_active(manager):
    mock_observer = mock.Mock()
    mock_observer.is_alive.return_value = True
    manager.file_observer = mock_observer
    manager.stop_file_monitoring()
    mock_observer.stop.assert_called_once()
    mock_observer.join.assert_called_once()

def test_stop_file_monitoring_when_inactive(manager, capsys):
    mock_observer = mock.Mock()
    mock_observer.is_alive.return_value = False
    manager.file_observer = mock_observer
    manager.stop_file_monitoring()
    captured = capsys.readouterr()
    assert "No active file monitoring to stop." in captured.out

def test_query_knowledge_delegates_to_rag_store(manager):
    mock_rag_store = manager.rag_store
    mock_rag_store.query.return_value = ["result1", "result2"]
    result = manager.query_knowledge("test query", n_results=2, filter_metadata={"foo": "bar"})
    mock_rag_store.query.assert_called_once_with(
        query_text="test query", n_results=2, filter_metadata={"foo": "bar"}
    )
    assert result == ["result1", "result2"]