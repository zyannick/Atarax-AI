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
    new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
    new_file.assert_called_once_with("foo.txt", mock_manifest, mock_rag_store, mock.ANY)
    mod_file.assert_not_called()
    del_file.assert_not_called()

def test_worker_modified_event(mock_manifest, mock_rag_store, chunk_config, mocker):
    tasks = [{"event_type": "modified", "path": "bar.txt"}]
    new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
    mod_file.assert_called_once_with("bar.txt", mock_manifest, mock_rag_store, mock.ANY)
    new_file.assert_not_called()
    del_file.assert_not_called()

# def test_worker_deleted_event(mock_manifest, mock_rag_store, chunk_config, mocker):
#     tasks = [{"event_type": "deleted", "path": "baz.txt"}]
#     new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
#     del_file.assert_called_once_with("baz.txt", mock_manifest, mock_rag_store)
#     new_file.assert_not_called()
#     mod_file.assert_not_called()

# def test_worker_moved_event(mock_manifest, mock_rag_store, chunk_config, mocker):
#     tasks = [{"event_type": "moved", "path": "old.txt", "dest_path": "new.txt"}]
#     new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
#     del_file.assert_called_once_with("old.txt", mock_manifest, mock_rag_store)
#     new_file.assert_called_once_with("new.txt", mock_manifest, mock_rag_store, mock.ANY)
#     mod_file.assert_not_called()

# def test_worker_moved_event_missing_dest_path(mock_manifest, mock_rag_store, chunk_config, mocker, capsys):
#     tasks = [{"event_type": "moved", "path": "old.txt"}]
#     new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
#     captured = capsys.readouterr()
#     assert "missing dest_path" in captured.out
#     new_file.assert_not_called()
#     mod_file.assert_not_called()
#     del_file.assert_not_called()

# def test_worker_unknown_event_type(mock_manifest, mock_rag_store, chunk_config, mocker, capsys):
#     tasks = [{"event_type": "unknown", "path": "foo.txt"}]
#     new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
#     captured = capsys.readouterr()
#     assert "Unknown event type" in captured.out
#     new_file.assert_not_called()
#     mod_file.assert_not_called()
#     del_file.assert_not_called()

# def test_worker_missing_path_in_task(mock_manifest, mock_rag_store, chunk_config, mocker, capsys):
#     tasks = [{"event_type": "created"}]
#     new_file, mod_file, del_file = run_worker_with_tasks(tasks, mock_manifest, mock_rag_store, chunk_config, mocker)
#     captured = capsys.readouterr()
#     assert "missing path" in captured.out
#     new_file.assert_not_called()
#     mod_file.assert_not_called()
#     del_file.assert_not_called()

# def test_worker_handles_queue_empty(mocker, mock_manifest, mock_rag_store, chunk_config):
#     q = mock.Mock()
#     q.get.side_effect = [queue.Empty(), None]
#     q.task_done = mock.Mock()

#     mocker.setattr(rag_updater, "SmartChunker", mock.Mock())
#     mocker.patch.object(rag_updater, "process_new_file", autospec=True)
#     mocker.patch.object(rag_updater, "process_modified_file", autospec=True)
#     mocker.patch.object(rag_updater, "process_deleted_file", autospec=True)

#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, chunk_config)
#     assert q.get.call_count == 2
    
    
# @pytest.fixture
# def mock_file(tmp_path, mocker):
#     file_path = tmp_path / "mod.txt"
#     file_path.write_text("hello world")
#     stat = file_path.stat()
#     mocker.setattr(rag_updater.Path, "exists", lambda self: self == file_path)
#     mocker.setattr(rag_updater.Path, "is_dir", lambda self: False)
#     return file_path

# @pytest.fixture
# def mock_manifest_entry():
#     return {
#         "hash": "abc123",
#         "timestamp": 1000,
#         "chunk_ids": ["id1", "id2"],
#         "status": "indexed"
#     }

# def test_modified_file_not_exists(mocker, mock_manifest, mock_rag_store):
#     mocker.setattr(rag_updater.Path, "exists", lambda self: False)
#     mocker.setattr(rag_updater.Path, "is_dir", lambda self: False)
#     deleted = mocker.patch.object(rag_updater, "process_deleted_file", autospec=True)
#     rag_updater.process_modified_file("nofile.txt", mock_manifest, mock_rag_store, mock.Mock())
#     deleted.assert_called_once_with("nofile.txt", mock_manifest, mock_rag_store)

# def test_modified_file_is_dir(mocker, mock_manifest, mock_rag_store):
#     mocker.setattr(rag_updater.Path, "exists", lambda self: True)
#     mocker.setattr(rag_updater.Path, "is_dir", lambda self: True)
#     deleted = mocker.patch.object(rag_updater, "process_deleted_file", autospec=True)
#     rag_updater.process_modified_file("adir", mock_manifest, mock_rag_store, mock.Mock())
#     deleted.assert_called_once_with("adir", mock_manifest, mock_rag_store)

# def test_modified_file_hash_none(mocker, mock_manifest, mock_rag_store, mock_file):
#     mocker.setattr(rag_updater, "get_file_hash", lambda path: None)
#     mocker.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=1234))
#     new_file = mocker.patch.object(rag_updater, "process_new_file", autospec=True)
#     rag_updater.process_modified_file(str(mock_file), mock_manifest, mock_rag_store, mock.Mock())
#     new_file.assert_not_called()

# def test_modified_file_hash_unchanged(mocker, mock_manifest, mock_rag_store, mock_file, mock_manifest_entry):
#     mocker.setattr(rag_updater, "get_file_hash", lambda path: "abc123")
#     mocker.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=2000))
#     mock_manifest.data = {str(mock_file): dict(mock_manifest_entry)}
#     rag_updater.process_modified_file(str(mock_file), mock_manifest, mock_rag_store, mock.Mock())
#     assert mock_manifest.data[str(mock_file)]["timestamp"] == 2000
#     mock_manifest.save.assert_called_once()

# def test_modified_file_hash_unchanged_timestamp_not_newer(mocker, mock_manifest, mock_rag_store, mock_file, mock_manifest_entry):
#     mocker.setattr(rag_updater, "get_file_hash", lambda path: "abc123")
#     mocker.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=1000))
#     mock_manifest.data = {str(mock_file): dict(mock_manifest_entry)}
#     rag_updater.process_modified_file(str(mock_file), mock_manifest, mock_rag_store, mock.Mock())
#     assert mock_manifest.data[str(mock_file)]["timestamp"] == 1000
#     mock_manifest.save.assert_not_called()

# def test_modified_file_hash_changed(mocker, mock_manifest, mock_rag_store, mock_file, mock_manifest_entry):
#     mocker.setattr(rag_updater, "get_file_hash", lambda path: "newhash")
#     mocker.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=3000))
#     mock_manifest.data = {str(mock_file): dict(mock_manifest_entry)}
#     new_file = mocker.patch.object(rag_updater, "process_new_file", autospec=True)
#     rag_updater.process_modified_file(str(mock_file), mock_manifest, mock_rag_store, mock.Mock())
#     mock_rag_store.delete_by_ids.assert_called_once_with(ids=mock_manifest_entry["chunk_ids"])
#     new_file.assert_called_once_with(str(mock_file), mock_manifest, mock_rag_store, mock.ANY)

# def test_modified_file_hash_changed_no_chunk_ids(mocker, mock_manifest, mock_rag_store, mock_file):
#     # Patch get_file_hash to return a new hash
#     mocker.setattr(rag_updater, "get_file_hash", lambda path: "newhash")
#     mocker.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=3000))
#     # Manifest entry without chunk_ids
#     mock_manifest.data = {str(mock_file): {"hash": "oldhash", "timestamp": 1000}}
#     new_file = mocker.patch.object(rag_updater, "process_new_file", autospec=True)
#     rag_updater.process_modified_file(str(mock_file), mock_manifest, mock_rag_store, mock.Mock())
#     mock_rag_store.delete_by_ids.assert_not_called()
#     new_file.assert_called_once_with(str(mock_file), mock_manifest, mock_rag_store, mock.ANY)

# def test_modified_file_exception(mocker, mock_manifest, mock_rag_store, mock_file, mock_manifest_entry):
#     # Patch get_file_hash to raise exception
#     def raise_exc(path):
#         raise RuntimeError("fail!")
#     mocker.setattr(rag_updater, "get_file_hash", raise_exc)
#     mock_manifest.data = {str(mock_file): dict(mock_manifest_entry)}
#     rag_updater.process_modified_file(str(mock_file), mock_manifest, mock_rag_store, mock.Mock())
#     # Should set status to error and call save
#     assert "error: fail!" in mock_manifest.data[str(mock_file)]["status"]
#     mock_manifest.save.assert_called_once()
    

# def test_process_deleted_file_with_chunk_ids(mocker, mock_manifest, mock_rag_store, capsys):
#     # Setup manifest entry with chunk_ids
#     mock_manifest.data = {"file1.txt": {"chunk_ids": ["id1", "id2"]}}
#     # Call function
#     rag_updater.process_deleted_file("file1.txt", mock_manifest, mock_rag_store)
#     # Should call delete_by_ids and save
#     mock_rag_store.delete_by_ids.assert_called_once_with(ids=["id1", "id2"])
#     mock_manifest.save.assert_called_once()
#     # Entry should be removed from manifest.data
#     assert "file1.txt" not in mock_manifest.data
#     out = capsys.readouterr().out
#     assert "Processing DELETED file: file1.txt" in out
#     assert "Deleting chunks for file1.txt from RAG store." in out
#     assert "Successfully processed deletion of file1.txt" in out

# def test_process_deleted_file_no_chunk_ids(mocker, mock_manifest, mock_rag_store, capsys):
#     # Setup manifest entry without chunk_ids
#     mock_manifest.data = {"file2.txt": {"foo": "bar"}}
#     rag_updater.process_deleted_file("file2.txt", mock_manifest, mock_rag_store)
#     # Should not call delete_by_ids or save
#     mock_rag_store.delete_by_ids.assert_not_called()
#     mock_manifest.save.assert_not_called()
#     # Entry should be removed from manifest.data
#     assert "file2.txt" not in mock_manifest.data
#     out = capsys.readouterr().out
#     assert "File file2.txt not found in manifest or no chunk IDs to delete." in out

# def test_process_deleted_file_not_in_manifest(mocker, mock_manifest, mock_rag_store, capsys):
#     # No entry in manifest
#     mock_manifest.data = {}
#     rag_updater.process_deleted_file("file3.txt", mock_manifest, mock_rag_store)
#     # Should not call delete_by_ids or save
#     mock_rag_store.delete_by_ids.assert_not_called()
#     mock_manifest.save.assert_not_called()
#     out = capsys.readouterr().out
#     assert "File file3.txt not found in manifest or no chunk IDs to delete." in out

# def test_process_deleted_file_exception(mocker, mock_manifest, mock_rag_store, capsys):
#     # Setup manifest entry with chunk_ids
#     mock_manifest.data = {"file4.txt": {"chunk_ids": ["id1"]}}
#     # Make delete_by_ids raise exception
#     mock_rag_store.delete_by_ids.side_effect = RuntimeError("fail delete")
#     rag_updater.process_deleted_file("file4.txt", mock_manifest, mock_rag_store)
#     out = capsys.readouterr().out
#     assert "Error processing deleted file file4.txt: fail delete" in out




# @pytest.fixture
# def patch_smart_chunker(mocker):
#     chunker = mock.Mock()
#     mocker.setattr(rag_updater, "SmartChunker", mock.Mock(return_value=chunker))
#     return chunker

# def test_worker_calls_process_new_file(mocker, mock_manifest, mock_rag_store, mock_chunker, patch_smart_chunker):
#     q = queue.Queue()
#     q.put({"event_type": "created", "path": "file.txt"})
#     q.put(None)
#     called = {}
#     def fake_new(path, manifest, rag_store, chunker):
#         called["new"] = (path, manifest, rag_store, chunker)
#     mocker.setattr(rag_updater, "process_new_file", fake_new)
#     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     assert called["new"][0] == "file.txt"
#     assert called["new"][1] is mock_manifest
#     assert called["new"][2] is mock_rag_store
#     assert called["new"][3] is patch_smart_chunker

# def test_worker_calls_process_modified_file(mocker, mock_manifest, mock_rag_store, mock_chunker, patch_smart_chunker):
#     q = queue.Queue()
#     q.put({"event_type": "modified", "path": "file2.txt"})
#     q.put(None)
#     called = {}
#     def fake_mod(path, manifest, rag_store, chunker):
#         called["mod"] = (path, manifest, rag_store, chunker)
#     mocker.setattr(rag_updater, "process_new_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_modified_file", fake_mod)
#     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     assert called["mod"][0] == "file2.txt"
#     assert called["mod"][1] is mock_manifest
#     assert called["mod"][2] is mock_rag_store
#     assert called["mod"][3] is patch_smart_chunker

# def test_worker_calls_process_deleted_file(mocker, mock_manifest, mock_rag_store, mock_chunker, patch_smart_chunker):
#     q = queue.Queue()
#     q.put({"event_type": "deleted", "path": "file3.txt"})
#     q.put(None)
#     called = {}
#     def fake_del(path, manifest, rag_store):
#         called["del"] = (path, manifest, rag_store)
#     mocker.setattr(rag_updater, "process_new_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_deleted_file", fake_del)
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     assert called["del"][0] == "file3.txt"
#     assert called["del"][1] is mock_manifest
#     assert called["del"][2] is mock_rag_store

# def test_worker_calls_process_moved_file(mocker, mock_manifest, mock_rag_store, mock_chunker, patch_smart_chunker):
#     q = queue.Queue()
#     q.put({"event_type": "moved", "path": "old.txt", "dest_path": "new.txt"})
#     q.put(None)
#     called = {}
#     def fake_del(path, manifest, rag_store):
#         called["del"] = (path, manifest, rag_store)
#     def fake_new(path, manifest, rag_store, chunker):
#         called["new"] = (path, manifest, rag_store, chunker)
#     mocker.setattr(rag_updater, "process_new_file", fake_new)
#     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_deleted_file", fake_del)
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     assert called["del"][0] == "old.txt"
#     assert called["new"][0] == "new.txt"

# def test_worker_moved_missing_dest_path(mocker, mock_manifest, mock_rag_store, mock_chunker, capsys, patch_smart_chunker):
#     q = queue.Queue()
#     q.put({"event_type": "moved", "path": "old.txt"})
#     q.put(None)
#     mocker.setattr(rag_updater, "process_new_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     out = capsys.readouterr().out
#     assert "missing dest_path" in out

# def test_worker_unknown_event_type(mocker, mock_manifest, mock_rag_store, mock_chunker, capsys, patch_smart_chunker):
#     q = queue.Queue()
#     q.put({"event_type": "foobar", "path": "file.txt"})
#     q.put(None)
#     mocker.setattr(rag_updater, "process_new_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     out = capsys.readouterr().out
#     assert "Unknown event type" in out

# # def test_worker_missing_path(mocker, mock_manifest, mock_rag_store, mock_chunker, capsys, patch_smart_chunker):
# #     q = queue.Queue()
# #     q.put({"event_type": "created"})
# #     q.put(None)
# #     mocker.setattr(rag_updater, "process_new_file", mock.Mock())
# #     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
# #     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
# #     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
# #     out = capsys.readouterr().out
# #     assert "missing path" in out

# def test_worker_queue_empty(mocker, mock_manifest, mock_rag_store, mock_chunker, patch_smart_chunker):
#     q = mock.Mock()
#     q.get.side_effect = [queue.Empty(), None]
#     q.task_done = mock.Mock()
#     mocker.setattr(rag_updater, "process_new_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
#     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
#     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
#     assert q.get.call_count == 2

# # def test_worker_unhandled_exception(mocker, mock_manifest, mock_rag_store, mock_chunker, capsys, patch_smart_chunker):
# #     q = queue.Queue()
# #     q.put({"event_type": "created", "path": "file.txt"})
# #     q.put(None)
# #     def raise_exc(*a, **kw):
# #         raise RuntimeError("fail!")
# #     mocker.setattr(rag_updater, "process_new_file", raise_exc)
# #     mocker.setattr(rag_updater, "process_modified_file", mock.Mock())
# #     mocker.setattr(rag_updater, "process_deleted_file", mock.Mock())
# #     rag_updater.rag_update_worker(q, mock_manifest, mock_rag_store, mock_chunker)
# #     out = capsys.readouterr().out
# #     assert "Unhandled error processing task" in out
# #     assert "fail!" in out



# @pytest.fixture
# def patch_set_base_metadata(mocker):
#     meta = {"file_timestamp": 1234567890, "file_hash": "abcdef1234567890"}
#     mocker.setattr(rag_updater, "set_base_metadata", lambda path: dict(meta))
#     return meta

# def test_process_new_file_success(mocker, mock_manifest, mock_rag_store, mock_chunker, dummy_file_path, patch_set_base_metadata, capsys):
#     # Setup chunker to return chunks
#     chunks = [
#         DummyChunk("chunk1", {"meta1": "val1"}),
#         DummyChunk("chunk2", {"meta2": "val2"}),
#     ]
#     mock_chunker.ingest_file.return_value = chunks

#     rag_updater.process_new_file(dummy_file_path, mock_manifest, mock_rag_store, mock_chunker)

#     # Should call add_chunks with correct ids, texts, metadatas
#     expected_ids = [
#         f"{dummy_file_path}_abcdef12_chunk_0",
#         f"{dummy_file_path}_abcdef12_chunk_1",
#     ]
#     mock_rag_store.add_chunks.assert_called_once_with(
#         ids=expected_ids,
#         texts=["chunk1", "chunk2"],
#         metadatas=["val1", "val2"],
#     )
#     mock_manifest.add_file.assert_called_once()
#     mock_manifest.save.assert_called_once()
#     out = capsys.readouterr().out
#     assert "WORKER: Processing NEW file" in out
#     assert "WORKER: Adding 2 chunks" in out
#     assert "WORKER: Successfully processed and indexed" in out

# def test_process_new_file_no_chunks(mocker, mock_manifest, mock_rag_store, mock_chunker, dummy_file_path, patch_set_base_metadata, capsys):
#     mock_chunker.ingest_file.return_value = []
#     rag_updater.process_new_file(dummy_file_path, mock_manifest, mock_rag_store, mock_chunker)
#     mock_rag_store.add_chunks.assert_not_called()
#     mock_manifest.add_file.assert_not_called()
#     mock_manifest.save.assert_not_called()
#     out = capsys.readouterr().out
#     assert "No chunks generated" in out

# def test_process_new_file_exception(mocker, mock_manifest, mock_rag_store, mock_chunker, dummy_file_path, patch_set_base_metadata, capsys):
#     chunks = [
#         DummyChunk("chunk1", {"meta1": "val1"}),
#     ]
#     mock_chunker.ingest_file.return_value = chunks
#     mock_rag_store.add_chunks.side_effect = RuntimeError("fail add")
#     mock_manifest.data = {dummy_file_path: {"status": "indexed"}}
#     mock_manifest.add_file.side_effect = lambda *a, **kw: None

#     rag_updater.process_new_file(dummy_file_path, mock_manifest, mock_rag_store, mock_chunker)
#     out = capsys.readouterr().out
#     assert "Error processing new file" in out
#     assert "fail add" in out
#     assert "error: fail add" in mock_manifest.data[dummy_file_path]["status"]
#     mock_manifest.save.assert_called()

# def test_process_new_file_exception_no_manifest_entry(mocker, mock_manifest, mock_rag_store, mock_chunker, dummy_file_path, patch_set_base_metadata, capsys):
#     # Setup chunker to return chunks
#     chunks = [
#         DummyChunk("chunk1", {"meta1": "val1"}),
#     ]
#     mock_chunker.ingest_file.return_value = chunks
#     mock_rag_store.add_chunks.side_effect = RuntimeError("fail add")
#     # Manifest.data has no entry for file
#     mock_manifest.data = {}
#     mock_manifest.add_file.side_effect = lambda *a, **kw: None

#     rag_updater.process_new_file(dummy_file_path, mock_manifest, mock_rag_store, mock_chunker)
#     out = capsys.readouterr().out
#     assert "Error processing new file" in out
#     # Should not raise, should not call save (since no manifest entry)
#     mock_manifest.save.assert_not_called()



