from typing import Any, Dict
import pytest
import asyncio
from unittest import mock
from ataraxai.praxis.modules.rag import rag_updater

@pytest.mark.asyncio
async def test_process_new_file_success(monkeypatch: mock.MagicMock):
    file_path_str = "/tmp/test.txt"
    fake_file_hash = "abc123456789"
    fake_timestamp = 1234567890.0

    manifest = mock.MagicMock()
    rag_store = mock.MagicMock()
    chunker = mock.MagicMock()
    logger = mock.MagicMock()

    monkeypatch.setattr(
        rag_updater, "set_base_metadata",
        lambda fp: {"file_hash": fake_file_hash, "file_timestamp": fake_timestamp}
    )

    chunk = mock.MagicMock()
    chunk.content = "chunk content"
    chunk.metadata = {"meta": "data"}
    chunker.ingest_file.return_value = [chunk, chunk]

    await rag_updater.process_new_file(file_path_str, manifest, rag_store, chunker, logger)

    assert rag_store.add_chunks.called, f"Expected add_chunks to be called, but it was not. Call args: {rag_store.add_chunks.call_args}"
    assert manifest.add_file.called, f"Expected add_file to be called, but it was not. Call args: {manifest.add_file.call_args}"
    assert manifest.save.called, f"Expected manifest.save to be called, but it was not. Call args: {manifest.save.call_args}"

@pytest.mark.asyncio
async def test_process_new_file_no_chunks(monkeypatch : mock.MagicMock):
    file_path_str = "/tmp/test.txt"
    manifest = mock.MagicMock()
    rag_store = mock.MagicMock()
    chunker = mock.MagicMock()
    logger = mock.MagicMock()

    monkeypatch.setattr(
        rag_updater, "set_base_metadata",
        lambda fp: {"file_hash": "hash", "file_timestamp": 1}
    )
    chunker.ingest_file.return_value = []

    await rag_updater.process_new_file(file_path_str, manifest, rag_store, chunker, logger)
    assert not rag_store.add_chunks.called
    assert not manifest.add_file.called

@pytest.mark.asyncio
async def test_process_new_file_exception(monkeypatch : mock.MagicMock):
    file_path_str = "/tmp/test.txt"
    manifest = mock.MagicMock()
    rag_store = mock.MagicMock()
    chunker = mock.MagicMock()
    logger = mock.MagicMock()

    monkeypatch.setattr(
        rag_updater, "set_base_metadata",
        lambda fp: {"file_hash": "hash", "file_timestamp": 1}
    )
    chunk = mock.MagicMock()
    chunk.content = "chunk"
    chunk.metadata = {}
    chunker.ingest_file.return_value = [chunk]
    rag_store.add_chunks.side_effect = Exception("fail")

    manifest.data = {file_path_str: {}}

    await rag_updater.process_new_file(file_path_str, manifest, rag_store, chunker, logger)
    assert manifest.save.called

@pytest.mark.asyncio
async def test_process_modified_file_hash_unchanged(monkeypatch : mock.MagicMock):
    file_path_str = "/tmp/test.txt"
    manifest = mock.MagicMock()
    rag_store = mock.MagicMock()
    chunker = mock.MagicMock()
    logger = mock.MagicMock()

    # Patch Path.exists and Path.is_dir
    monkeypatch.setattr(rag_updater.Path, "exists", lambda self: True)
    monkeypatch.setattr(rag_updater.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=2))

    monkeypatch.setattr(rag_updater, "get_file_hash", lambda fp: "hash")
    manifest.data = {file_path_str: {"hash": "hash", "timestamp": 1}}

    await rag_updater.process_modified_file(file_path_str, manifest, rag_store, chunker, logger)
    assert manifest.save.called

@pytest.mark.asyncio
async def test_process_modified_file_hash_changed(monkeypatch : mock.MagicMock):
    file_path_str = "/tmp/test.txt"
    manifest = mock.MagicMock()
    rag_store = mock.MagicMock()
    chunker = mock.MagicMock()
    logger = mock.MagicMock()

    monkeypatch.setattr(rag_updater.Path, "exists", lambda self: True)
    monkeypatch.setattr(rag_updater.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(rag_updater.Path, "stat", lambda self: mock.Mock(st_mtime=2))
    monkeypatch.setattr(rag_updater, "get_file_hash", lambda fp: "newhash")

    manifest.data = {file_path_str: {"hash": "oldhash", "chunk_ids": ["id1"]}}

    called = {}

    async def fake_process_new_file(*args, **kwargs):
        called["called"] = True

    monkeypatch.setattr(rag_updater, "process_new_file", fake_process_new_file)

    await rag_updater.process_modified_file(file_path_str, manifest, rag_store, chunker, logger)
    assert rag_store.delete_by_ids.called
    assert called.get("called")

# @pytest.mark.asyncio
# async def test_process_modified_file_deleted(monkeypatch : mock.MagicMock):
#     file_path_str = "/tmp/test.txt"
#     manifest = mock.MagicMock()
#     rag_store = mock.MagicMock()
#     chunker = mock.MagicMock()
#     logger = mock.MagicMock()

#     monkeypatch.setattr(rag_updater.Path, "exists", lambda self: False)
#     called = {}

#     async def fake_process_deleted_file(*args, **kwargs):
#         called["called"] = True

#     monkeypatch.setattr(rag_updater, "process_deleted_file", fake_process_deleted_file)

#     await rag_updater.process_modified_file(file_path_str, manifest, rag_store, chunker, logger)
#     assert called.get("called")

# @pytest.mark.asyncio
# async def test_process_deleted_file_success():
#     file_path_str = "/tmp/test.txt"
#     manifest = mock.MagicMock()
#     rag_store = mock.MagicMock()
#     logger = mock.MagicMock()
#     manifest.data = {file_path_str: {"chunk_ids": ["id1", "id2"]}}

#     await rag_updater.process_deleted_file(file_path_str, manifest, rag_store, logger)
#     assert rag_store.delete_by_ids.called
#     assert manifest.save.called
#     assert file_path_str not in manifest.data

# @pytest.mark.asyncio
# async def test_process_deleted_file_no_entry():
#     file_path_str = "/tmp/test.txt"
#     manifest = mock.MagicMock()
#     rag_store = mock.MagicMock()
#     logger = mock.MagicMock()
#     manifest.data = {}

#     await rag_updater.process_deleted_file(file_path_str, manifest, rag_store, logger)
#     assert not rag_store.delete_by_ids.called

# @pytest.mark.asyncio
# async def test_rag_update_worker_async_created(monkeypatch : mock.MagicMock):
#     manifest = mock.MagicMock()
#     rag_store = mock.MagicMock()
#     logger = mock.MagicMock()
#     chunk_config = {}

#     queue :  asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
#     await queue.put({"event_type": "created", "path": "/tmp/test.txt"})
#     await queue.put({"event_type": "stop"})

#     called = {}

#     async def fake_process_new_file(*args, **kwargs):
#         called["created"] = True

#     monkeypatch.setattr(rag_updater, "process_new_file", fake_process_new_file)

#     await rag_updater.rag_update_worker_async(queue, manifest, rag_store, chunk_config, logger)
#     assert called.get("created")

# @pytest.mark.asyncio
# async def test_rag_update_worker_async_moved(monkeypatch : mock.MagicMock):
#     manifest = mock.MagicMock()
#     rag_store = mock.MagicMock()
#     logger = mock.MagicMock()
#     chunk_config = {}

#     queue :  asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
#     await queue.put({"event_type": "moved", "path": "/tmp/old.txt", "dest_path": "/tmp/new.txt"})
#     await queue.put({"event_type": "stop"})

#     called = {}

#     async def fake_process_deleted_file(*args, **kwargs):
#         called["deleted"] = True

#     async def fake_process_new_file(*args, **kwargs):
#         called["new"] = True

#     monkeypatch.setattr(rag_updater, "process_deleted_file", fake_process_deleted_file)
#     monkeypatch.setattr(rag_updater, "process_new_file", fake_process_new_file)

#     await rag_updater.rag_update_worker_async(queue, manifest, rag_store, chunk_config, logger)
#     assert called.get("deleted")
#     assert called.get("new")