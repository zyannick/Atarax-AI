import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

with patch("ataraxai.app_logic.modules.rag.resilient_indexer.chromadb") as mock_chromadb:
    from ataraxai.app_logic.modules.rag.resilient_indexer import (
        ResilientFileIndexer,
        start_rag_file_monitoring,
    )

class DummyEvent:
    def __init__(self, src_path, is_directory=False, dest_path=None):
        self.src_path = src_path
        self.is_directory = is_directory
        self.dest_path = dest_path

@pytest.fixture
def file_indexer():
    mock_manifest = MagicMock()
    mock_collection = MagicMock()
    return ResilientFileIndexer(mock_manifest, mock_collection)

def test_on_created_puts_task(file_indexer):
    event = DummyEvent("/tmp/test.txt")
    file_indexer.on_created(event)
    task = file_indexer.processing_queue.get_nowait()
    assert task["event_type"] == "created"
    assert task["file_path"] == "/tmp/test.txt"

def test_on_modified_puts_task(file_indexer):
    event = DummyEvent("/tmp/test2.txt")
    file_indexer.on_modified(event)
    task = file_indexer.processing_queue.get_nowait()
    assert task["event_type"] == "modified"
    assert task["file_path"] == "/tmp/test2.txt"

def test_on_deleted_puts_task(file_indexer):
    event = DummyEvent("/tmp/test3.txt")
    file_indexer.on_deleted(event)
    task = file_indexer.processing_queue.get_nowait()
    assert task["event_type"] == "deleted"
    assert task["file_path"] == "/tmp/test3.txt"

def test_on_moved_puts_task(file_indexer):
    event = DummyEvent("/tmp/old.txt", dest_path="/tmp/new.txt")
    file_indexer.on_moved(event)
    task = file_indexer.processing_queue.get_nowait()
    assert task["event_type"] == "moved"
    assert task["src_path"] == "/tmp/old.txt"
    assert task["dest_path"] == "/tmp/new.txt"

def test_directory_events_are_ignored(file_indexer):
    event = DummyEvent("/tmp/dir", is_directory=True)
    file_indexer.on_created(event)
    file_indexer.on_modified(event)
    file_indexer.on_deleted(event)
    file_indexer.on_moved(event)
    assert file_indexer.processing_queue.empty()

@patch("ataraxai.app_logic.modules.rag.resilient_indexer.Observer")
def test_start_rag_file_monitoring_watches_existing_paths(MockObserver):
    mock_manifest = MagicMock()
    mock_collection = MagicMock()
    mock_observer = MockObserver.return_value
    existing_path = "/tmp"
    with patch.object(Path, "exists", return_value=True):
        observer = start_rag_file_monitoring(
            [existing_path], mock_manifest, mock_collection
        )
        mock_observer.schedule.assert_called()
        assert observer == mock_observer
        mock_observer.start.assert_called_once()

@patch("ataraxai.app_logic.modules.rag.resilient_indexer.Observer")
def test_start_rag_file_monitoring_warns_on_missing_path(MockObserver):
    mock_manifest = MagicMock()
    mock_collection = MagicMock()
    with patch.object(Path, "exists", return_value=False):
        observer = start_rag_file_monitoring(
            ["/nonexistent"], mock_manifest, mock_collection
        )
        assert MockObserver.return_value.start.called

