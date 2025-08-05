import pytest
from unittest.mock import MagicMock, patch
from unittest import mock
from pathlib import Path
from unittest.mock import MagicMock
from queue import Queue
from ataraxai.praxis.modules.rag.resilient_indexer import ResilientFileIndexer
from ataraxai.praxis.modules.rag import resilient_indexer

class DummyEvent:
    def __init__(self, src_path, is_directory=False, dest_path=None):
        self.src_path = src_path
        self.is_directory = is_directory
        self.dest_path = dest_path

@pytest.fixture
def processing_queue():
    return Queue()

@pytest.fixture
def indexer(processing_queue):
    return ResilientFileIndexer(processing_queue)

def test_on_created_file(indexer, processing_queue):
    event = DummyEvent("/tmp/test.txt", is_directory=False)
    indexer.on_created(event)
    task = processing_queue.get_nowait()
    assert task == {"event_type": "created", "file_path": "/tmp/test.txt"}

def test_on_created_directory(indexer, processing_queue):
    event = DummyEvent("/tmp/dir", is_directory=True)
    indexer.on_created(event)
    assert processing_queue.empty()

def test_on_modified_file(indexer, processing_queue):
    event = DummyEvent("/tmp/test.txt", is_directory=False)
    indexer.on_modified(event)
    task = processing_queue.get_nowait()
    assert task == {"event_type": "modified", "file_path": "/tmp/test.txt"}

def test_on_modified_directory(indexer, processing_queue):
    event = DummyEvent("/tmp/dir", is_directory=True)
    indexer.on_modified(event)
    assert processing_queue.empty()

def test_on_deleted_file(indexer, processing_queue):
    event = DummyEvent("/tmp/test.txt", is_directory=False)
    indexer.on_deleted(event)
    task = processing_queue.get_nowait()
    assert task == {"event_type": "deleted", "file_path": "/tmp/test.txt"}

def test_on_deleted_directory(indexer, processing_queue):
    event = DummyEvent("/tmp/dir", is_directory=True)
    indexer.on_deleted(event)
    assert processing_queue.empty()

def test_on_moved_file(indexer, processing_queue):
    event = DummyEvent("/tmp/old.txt", is_directory=False, dest_path="/tmp/new.txt")
    indexer.on_moved(event)
    task = processing_queue.get_nowait()
    assert task == {
        "event_type": "moved",
        "src_path": "/tmp/old.txt",
        "dest_path": "/tmp/new.txt",
    }

def test_on_moved_directory(indexer, processing_queue):
    event = DummyEvent("/tmp/olddir", is_directory=True, dest_path="/tmp/newdir")
    indexer.on_moved(event)
    assert processing_queue.empty()
    

@patch("ataraxai.praxis.modules.rag.resilient_indexer.Observer")
@patch("ataraxai.praxis.modules.rag.resilient_indexer.ResilientFileIndexer")
@patch("ataraxai.praxis.modules.rag.resilient_indexer.threading.Thread")
def test_start_rag_file_monitoring_existing_paths(
    mock_thread, mock_indexer, mock_observer
):

    mock_manifest = MagicMock()
    mock_rag_store = MagicMock()
    chunk_config = {"chunk_size": 100}
    mock_obs_instance = MagicMock()
    mock_observer.return_value = mock_obs_instance
    mock_indexer_instance = MagicMock()
    mock_indexer.return_value = mock_indexer_instance
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance

    with patch("ataraxai.praxis.modules.rag.resilient_indexer.Path.exists", return_value=True):
        result = resilient_indexer.start_rag_file_monitoring(
            ["/tmp/dir1", "/tmp/dir2"], mock_manifest, mock_rag_store, chunk_config
        )

    mock_thread_instance.start.assert_called_once()
    assert mock_obs_instance.schedule.call_count == 2
    mock_obs_instance.start.assert_called_once()
    assert result == mock_obs_instance

@patch("ataraxai.praxis.modules.rag.resilient_indexer.Observer")
@patch("ataraxai.praxis.modules.rag.resilient_indexer.ResilientFileIndexer")
@patch("ataraxai.praxis.modules.rag.resilient_indexer.threading.Thread")
def test_start_rag_file_monitoring_nonexistent_paths(
    mock_thread, mock_indexer, mock_observer, capsys
):

    mock_manifest = MagicMock()
    mock_rag_store = MagicMock()
    chunk_config = {"chunk_size": 100}
    mock_obs_instance = MagicMock()
    mock_observer.return_value = mock_obs_instance
    mock_indexer_instance = MagicMock()
    mock_indexer.return_value = mock_indexer_instance
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance

    with patch("ataraxai.praxis.modules.rag.resilient_indexer.Path.exists", return_value=False):
        result = resilient_indexer.start_rag_file_monitoring(
            ["/notfound/dir1", "/notfound/dir2"], mock_manifest, mock_rag_store, chunk_config
        )

    mock_thread_instance.start.assert_called_once()
    mock_obs_instance.schedule.assert_not_called()
    mock_obs_instance.start.assert_called_once()
    assert result == mock_obs_instance

    captured = capsys.readouterr()
    assert "Warning: Path not found, cannot watch: /notfound/dir1" in captured.out
    assert "Warning: Path not found, cannot watch: /notfound/dir2" in captured.out

@patch("ataraxai.praxis.modules.rag.resilient_indexer.Observer")
@patch("ataraxai.praxis.modules.rag.resilient_indexer.ResilientFileIndexer")
@patch("ataraxai.praxis.modules.rag.resilient_indexer.threading.Thread")
def test_start_rag_file_monitoring_mixed_paths(
    mock_thread, mock_indexer, mock_observer, capsys
):

    mock_manifest = MagicMock()
    mock_rag_store = MagicMock()
    chunk_config = {"chunk_size": 100}
    mock_obs_instance = MagicMock()
    mock_observer.return_value = mock_obs_instance
    mock_indexer_instance = MagicMock()
    mock_indexer.return_value = mock_indexer_instance
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance

    def exists_side_effect(self):
        return str(self) == "/exists/dir"

    with patch("ataraxai.praxis.modules.rag.resilient_indexer.Path.exists", new=exists_side_effect):
        result = resilient_indexer.start_rag_file_monitoring(
            ["/exists/dir", "/notfound/dir"], mock_manifest, mock_rag_store, chunk_config
        )

    mock_thread_instance.start.assert_called_once()
    mock_obs_instance.schedule.assert_called_once()
    mock_obs_instance.start.assert_called_once()
    assert result == mock_obs_instance

    captured = capsys.readouterr()
    assert "Watching directory: /exists/dir" in captured.out
    assert "Warning: Path not found, cannot watch: /notfound/dir" in captured.out



