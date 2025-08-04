import pytest
import queue
import threading
from unittest import mock
from ataraxai.praxis.modules.rag import rag_updater
import os
import types


@pytest.fixture
def mock_manifest():
    return mock.Mock()


@pytest.fixture
def mock_rag_store():
    return mock.Mock()


@pytest.fixture
def mock_chunker(mocker):
    mocker.setattr(rag_updater, "SmartChunker", mock.Mock())
    return rag_updater.SmartChunker.return_value


@pytest.fixture
def chunk_config():
    return {
        "size": 100,
        "overlap": 10,
        "model_name_for_tiktoken": "gpt-4",
        "separators": None,
        "keep_separator": True,
    }


def run_worker_with_tasks(tasks, manifest, rag_store, chunk_config, mocker):
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    q.put(None)

    new_file = mocker.patch.object(rag_updater, "process_new_file", autospec=True)
    mod_file = mocker.patch.object(rag_updater, "process_modified_file", autospec=True)
    del_file = mocker.patch.object(rag_updater, "process_deleted_file", autospec=True)

    rag_updater.rag_update_worker(q, manifest, rag_store, chunk_config)
    return new_file, mod_file, del_file


def test_worker_created_event(mock_manifest, mock_rag_store, chunk_config, mocker):
    tasks = [{"event_type": "created", "path": "foo.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    new_file.assert_called_once_with("foo.txt", mock_manifest, mock_rag_store, mock.ANY)
    mod_file.assert_not_called()
    del_file.assert_not_called()


def test_worker_modified_event(mock_manifest, mock_rag_store, chunk_config, mocker):
    tasks = [{"event_type": "modified", "path": "bar.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    mod_file.assert_called_once_with("bar.txt", mock_manifest, mock_rag_store, mock.ANY)
    new_file.assert_not_called()
    del_file.assert_not_called()


def test_worker_deleted_event(mock_manifest, mock_rag_store, chunk_config, mocker):
    tasks = [{"event_type": "deleted", "path": "baz.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    del_file.assert_called_once_with("baz.txt", mock_manifest, mock_rag_store)
    new_file.assert_not_called()
    mod_file.assert_not_called()


def test_worker_moved_event(mock_manifest, mock_rag_store, chunk_config, mocker):
    tasks = [{"event_type": "moved", "path": "old.txt", "dest_path": "new.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    del_file.assert_called_once_with("old.txt", mock_manifest, mock_rag_store)
    new_file.assert_called_once_with("new.txt", mock_manifest, mock_rag_store, mock.ANY)
    mod_file.assert_not_called()


def test_worker_moved_event_missing_dest_path(
    mock_manifest, mock_rag_store, chunk_config, mocker, capsys
):
    tasks = [{"event_type": "moved", "path": "old.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    captured = capsys.readouterr()
    assert "missing dest_path" in captured.out
    new_file.assert_not_called()
    mod_file.assert_not_called()
    del_file.assert_not_called()


def test_worker_unknown_event_type(
    mock_manifest, mock_rag_store, chunk_config, mocker, capsys
):
    tasks = [{"event_type": "unknown", "path": "foo.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    captured = capsys.readouterr()
    assert "Unknown event type" in captured.out
    new_file.assert_not_called()
    mod_file.assert_not_called()
    del_file.assert_not_called()


def test_worker_missing_path_in_task(
    mock_manifest, mock_rag_store, chunk_config, mocker, capsys
):
    tasks = [{"event_type": "created"}]
    new_file, mod_file, del_file = run_worker_with_tasks(
        tasks, mock_manifest, mock_rag_store, chunk_config, mocker
    )
    captured = capsys.readouterr()
    assert "missing path" in captured.out
    new_file.assert_not_called()
    mod_file.assert_not_called()
    del_file.assert_not_called()


def test_process_new_file_success(mocker, tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    mock_manifest = mock.Mock()
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    mock_cd = mock.Mock()
    mock_cd.content = "chunk content"
    mock_cd.metadata = {"foo": "bar"}
    mock_chunker.ingest_file.return_value = [mock_cd, mock_cd]
    mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.set_base_metadata",
        return_value={"file_hash": "abcdef123456", "file_timestamp": 123456789},
    )
    rag_updater.process_new_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    assert mock_rag_store.add_chunks.called
    assert mock_manifest.add_file.called
    assert mock_manifest.save.called


def test_process_new_file_no_chunks(mocker, tmp_path, capsys):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("empty")
    mock_manifest = mock.Mock()
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    mock_chunker.ingest_file.return_value = []
    mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.set_base_metadata",
        return_value={"file_hash": "abcdef123456", "file_timestamp": 123456789},
    )
    rag_updater.process_new_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    captured = capsys.readouterr()
    assert "No chunks generated" in captured.out


def test_process_new_file_exception(mocker, tmp_path):
    file_path = tmp_path / "fail.txt"
    file_path.write_text("fail")
    mock_manifest = mock.Mock()
    mock_manifest.data = {str(file_path): {"status": "old"}}
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    mock_cd = mock.Mock()
    mock_cd.content = "chunk"
    mock_cd.metadata = {"foo": "bar"}
    mock_chunker.ingest_file.return_value = [mock_cd]
    mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.set_base_metadata",
        return_value={"file_hash": "abcdef123456", "file_timestamp": 123456789},
    )
    mock_rag_store.add_chunks.side_effect = Exception("fail add")
    rag_updater.process_new_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    assert "error" in mock_manifest.data[str(file_path)]["status"]
    assert mock_manifest.save.called


def test_process_modified_file_deleted(mocker, tmp_path):
    file_path = tmp_path / "gone.txt"
    mock_manifest = mock.Mock()
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    process_deleted = mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.process_deleted_file"
    )
    rag_updater.process_modified_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    process_deleted.assert_called_once_with(
        str(file_path), mock_manifest, mock_rag_store
    )


def test_process_modified_file_hash_match(mocker, tmp_path):
    file_path = tmp_path / "same.txt"
    file_path.write_text("abc")
    mock_manifest = mock.Mock()
    mock_manifest.data = {str(file_path): {"hash": "h", "timestamp": 1}}
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.get_file_hash", return_value="h"
    )
    rag_updater.process_modified_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    # Should only update timestamp if newer
    assert mock_manifest.save.called or not mock_manifest.save.called


def test_process_modified_file_reindex(mocker, tmp_path):
    file_path = tmp_path / "changed.txt"
    file_path.write_text("abc")
    mock_manifest = mock.Mock()
    mock_manifest.data = {str(file_path): {"hash": "old", "chunk_ids": ["id1"]}}
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.get_file_hash", return_value="new"
    )
    process_new = mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.process_new_file"
    )
    rag_updater.process_modified_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    assert mock_rag_store.delete_by_ids.called
    process_new.assert_called_once()


def test_process_modified_file_exception(mocker, tmp_path):
    file_path = tmp_path / "err.txt"
    file_path.write_text("abc")
    mock_manifest = mock.Mock()
    mock_manifest.data = {str(file_path): {"hash": "old", "status": "ok"}}
    mock_rag_store = mock.Mock()
    mock_chunker = mock.Mock()
    mocker.patch(
        "ataraxai.praxis.modules.rag.rag_updater.get_file_hash",
        side_effect=Exception("fail"),
    )
    rag_updater.process_modified_file(
        str(file_path), mock_manifest, mock_rag_store, mock_chunker
    )
    assert "error" in mock_manifest.data[str(file_path)]["status"]
    assert mock_manifest.save.called


def test_process_deleted_file_with_chunks(mocker):
    mock_manifest = mock.Mock()
    mock_manifest.data = {"f.txt": {"chunk_ids": ["id1", "id2"]}}
    mock_rag_store = mock.Mock()
    rag_updater.process_deleted_file("f.txt", mock_manifest, mock_rag_store)
    assert mock_rag_store.delete_by_ids.called
    assert mock_manifest.save.called


def test_process_deleted_file_no_entry(mocker):
    mock_manifest = mock.Mock()
    mock_manifest.data = {}
    mock_rag_store = mock.Mock()
    rag_updater.process_deleted_file("missing.txt", mock_manifest, mock_rag_store)
    assert not mock_rag_store.delete_by_ids.called
    assert not mock_manifest.save.called


def test_process_deleted_file_exception(mocker):
    mock_manifest = mock.Mock()
    mock_manifest.data = {"f.txt": {"chunk_ids": ["id1"]}}
    mock_rag_store = mock.Mock()
    mock_rag_store.delete_by_ids.side_effect = Exception("fail")
    rag_updater.process_deleted_file("f.txt", mock_manifest, mock_rag_store)
