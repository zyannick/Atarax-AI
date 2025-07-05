import pytest
from unittest import mock
from pathlib import Path
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager


@pytest.fixture
def mock_preferences_manager():
    """
    A mock that returns different values based on the preference key requested.
    """
    mock_pm = mock.Mock()

    def get_side_effect(key, default_value=None):
        if key == "rag_embedder_model":
            return "sentence-transformers/all-MiniLM-L6-v2"
        elif key == "n_result":
            return 5
        elif key == "n_result_final":
            return 3
        else:
            return default_value

    mock_pm.get.side_effect = get_side_effect

    return mock_pm


@pytest.fixture
def tmp_app_data_root(tmp_path):
    return tmp_path / "app_data"


@pytest.fixture
def manager(mock_preferences_manager, tmp_app_data_root):
    """
    This fixture now correctly mocks the class dependencies and injects
    a mock for the constructor argument.
    """
    # Create a mock object for the core_ai_service dependency
    mock_core_ai = mock.Mock()

    # Use mock.patch for dependencies that are imported and used
    # inside the AtaraxAIRAGManager module.
    with mock.patch(
        "ataraxai.app_logic.modules.rag.ataraxai_rag_manager.AtaraxAIEmbedder"
    ), mock.patch(
        "ataraxai.app_logic.modules.rag.ataraxai_rag_manager.RAGStore"
    ), mock.patch(
        "ataraxai.app_logic.modules.rag.ataraxai_rag_manager.RAGManifest"
    ):

        # Instantiate the manager, passing the mock_core_ai object directly
        # as the 'core_ai_service' argument.
        manager_instance = AtaraxAIRAGManager(
            preferences_manager_instance=mock_preferences_manager,
            app_data_root_path=tmp_app_data_root,
            core_ai_service=mock_core_ai,
        )
        yield manager_instance


def test_init_creates_paths_and_instances(manager, tmp_app_data_root):
    rag_store_db_path = tmp_app_data_root / "rag_chroma_store"
    manifest_file_path = tmp_app_data_root / "rag_manifest.json"
    assert rag_store_db_path.exists()
    assert isinstance(manager.app_data_root_path, Path)
    assert manager.manifest_file_path == manifest_file_path


def test_start_file_monitoring_starts_observer(manager):
    with mock.patch(
        "ataraxai.app_logic.modules.rag.ataraxai_rag_manager.start_rag_file_monitoring"
    ) as mock_start_monitor:
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
    mock_rag_store.query.return_value = {
        "documents": [["result1", "result2"]],
    }

    manager.use_hyde = False
    manager.rag_use_reranking = False
    
    result = manager.query_knowledge("test query", filter_metadata={"foo": "bar"})

    mock_rag_store.query.assert_called_once_with(
        query_text="test query",
        filter_metadata={"foo": "bar"},
        n_results=manager.n_result 
    )
    
    assert result == ["result1", "result2"]