import pytest
import tempfile
import shutil
import os
import json
from unittest import mock
from pathlib import Path
from datetime import datetime
from ataraxai.praxis.utils.app_directories import AppDirectories
import hashlib

from ataraxai.praxis.modules.models_manager.models_manager import (
    ModelsManager,
    LlamaCPPModelInfo,
    ModelDownloadStatus,
    ModelDownloadInfo,
)

class DummyLogger:
    def __init__(self):
        self.messages = []
    def info(self, msg): self.messages.append(('info', msg))
    def warning(self, msg): self.messages.append(('warning', msg))
    def error(self, msg): self.messages.append(('error', msg))

@pytest.fixture
def temp_dirs():
    tmp = tempfile.mkdtemp()
    dirs = mock.Mock(spec=AppDirectories)
    dirs.data = Path(tmp)
    yield dirs
    shutil.rmtree(tmp)

@pytest.fixture
def logger():
    return DummyLogger()

@pytest.fixture
def manager(temp_dirs, logger):
    return ModelsManager(temp_dirs, logger)

def test_manifest_load_and_save(manager):
    manager.manifest = {"models": [{"repo_id": "test", "filename": "file.gguf"}], "last_updated": None}
    manager._save_manifest()
    manager._load_manifest()
    assert "models" in manager.manifest
    assert isinstance(manager.manifest["models"], list)

def test_calculate_sha256(tmp_path, manager):
    file = tmp_path / "test.bin"
    file.write_bytes(b"abc123")
    hash_val = manager._calculate_sha256(file)
    expected = hashlib.sha256(b"abc123").hexdigest()
    assert hash_val == expected

def test_list_available_files(manager):
    with mock.patch.object(manager.hf_api, "list_repo_files", return_value=["a.gguf", "b.bin", "c.txt"]):
        files = manager.list_available_files("dummy/repo")
        assert "a.gguf" in files
        assert "b.bin" in files
        assert "c.txt" not in files

def test_search_models(manager):
    dummy_model = mock.Mock()
    dummy_model.id = "org/model"
    dummy_model.__dict__ = {"id": "org/model", "downloads": 10, "likes": 2, "created_at": datetime.now().isoformat()}
    with mock.patch.object(manager.hf_api, "list_models", return_value=[dummy_model]), \
         mock.patch.object(manager.hf_api, "list_repo_files", return_value=["model-Q4_K.gguf", "other.bin"]):
        results = manager.search_models("test")
        assert any(isinstance(m, LlamaCPPModelInfo) for m in results)
        assert any("Q4" in m.quantization_bit for m in results)

def test_add_to_manifest(manager):
    repo_id = "org/model"
    filename = "file.gguf"
    model_path = "/tmp/file.gguf"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id=repo_id,
        filename=filename,
        local_path=model_path,
        downloaded_at=datetime.now(),
        file_size=123,
        created_at=datetime.now(),
        downloads=1,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    manager._add_to_manifest(repo_id, filename, model_path, model_info)
    assert any(m["repo_id"] == repo_id and m["filename"] == filename for m in manager.manifest["models"])

def test_start_download_task_sets_status(manager):
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="org/model",
        filename="file.gguf",
        local_path="/tmp/file.gguf",
        downloaded_at=datetime.now(),
        file_size=123,
        created_at=datetime.now(),
        downloads=1,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    with mock.patch.object(manager, "_download_worker"):
        task_id = "task123"
        manager.start_download_task(task_id, model_info)
        assert task_id in manager._download_tasks
        info = manager._download_tasks[task_id]
        assert info.status == ModelDownloadStatus.STARTING
        assert info.repo_id == "org/model"
        assert info.filename == "file.gguf"

def test_progress_callback_wrapper_updates_task(manager):
    task_id = "task456"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="org/model",
        filename="file.gguf",
        local_path="/tmp/file.gguf",
        downloaded_at=datetime.now(),
        file_size=123,
        created_at=datetime.now(),
        downloads=1,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    manager._download_tasks[task_id] = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.STARTING,
        percentage=0.0,
        repo_id="org/model",
        filename="file.gguf",
        created_at=datetime.now(),
        message="Download task started.",
        model_info=model_info,
    )
    called = {}
    def user_callback(downloaded, total):
        called['downloaded'] = downloaded
        called['total'] = total

    cb = manager._progress_callback_wrapper(task_id, user_callback)
    cb(50, 100)
    info = manager._download_tasks[task_id]
    assert info.percentage == 0.5
    assert info.downloaded_bytes == 50
    assert info.file_size == 100
    assert called['downloaded'] == 50
    assert called['total'] == 100

def test_verify_file_integrity_passes_and_fails(manager, tmp_path):
    file = tmp_path / "test.bin"
    file.write_bytes(b"abc123")
    expected_hash = hashlib.sha256(b"abc123").hexdigest()
    with mock.patch.object(manager, "_get_expected_sha256", return_value=expected_hash):
        assert manager._verify_file_integrity(file, "repo", "test.bin") is True
    with mock.patch.object(manager, "_get_expected_sha256", return_value="wronghash"):
        assert manager._verify_file_integrity(file, "repo", "test.bin") is False
    with mock.patch.object(manager, "_get_expected_sha256", return_value=None):
        assert manager._verify_file_integrity(file, "repo", "test.bin") is True

def test_get_download_status_returns_info(manager):
    task_id = "task789"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="org/model",
        filename="file.gguf",
        local_path="/tmp/file.gguf",
        downloaded_at=datetime.now(),
        file_size=123,
        created_at=datetime.now(),
        downloads=1,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    info = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.DOWNLOADING,
        percentage=0.5,
        repo_id="org/model",
        filename="file.gguf",
        created_at=datetime.now(),
        message="Downloading...",
        model_info=model_info,
    )
    manager._download_tasks[task_id] = info
    result = manager.get_download_status(task_id)
    assert result is not None
    assert result["task_id"] == task_id
    assert result["status"] == ModelDownloadStatus.DOWNLOADING
    assert result["repo_id"] == "org/model"
    assert result["filename"] == "file.gguf"
    


def test_add_to_manifest_adds_and_updates(manager):

    repo_id = "org/model"
    filename = "file1.gguf"
    model_path = "/tmp/file1.gguf"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id=repo_id,
        filename=filename,
        local_path=model_path,
        downloaded_at=datetime.now(),
        file_size=100,
        created_at=datetime.now(),
        downloads=5,
        likes=2,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    manager.manifest = {"models": [], "last_updated": None}
    manager._add_to_manifest(repo_id, filename, model_path, model_info)
    assert any(m["repo_id"] == repo_id and m["filename"] == filename for m in manager.manifest["models"])

    model_info.file_size = 200
    manager._add_to_manifest(repo_id, filename, model_path, model_info)
    found = [m for m in manager.manifest["models"] if m["repo_id"] == repo_id and m["filename"] == filename]
    assert len(found) == 1
    assert found[0]["file_size"] == 200, f"Expected file size to be updated to 200, got {found[0]['file_size']}"

def test_download_worker_success_and_failure(manager, tmp_path):
    repo_id = "org/model"
    filename = "file.gguf"
    model_path = str(tmp_path / filename)
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id=repo_id,
        filename=filename,
        local_path=model_path,
        downloaded_at=datetime.now(),
        file_size=123,
        created_at=datetime.now(),
        downloads=1,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    task_id = "task001"
    manager._download_tasks[task_id] = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.STARTING,
        percentage=0.0,
        repo_id=repo_id,
        filename=filename,
        created_at=datetime.now(),
        message="Download task started.",
        model_info=model_info,
    )
    with mock.patch.object(manager, "_download_with_progress", return_value=model_path), \
            mock.patch.object(manager, "_verify_file_integrity", return_value=True), \
            mock.patch.object(manager, "_add_to_manifest"):
        manager._download_worker(task_id, repo_id, filename, model_info)
        assert manager._download_tasks[task_id].status == ModelDownloadStatus.COMPLETED

    manager._download_tasks[task_id].status = ModelDownloadStatus.STARTING
    with mock.patch.object(manager, "_download_with_progress", side_effect=Exception("fail")), \
            mock.patch.object(manager, "_verify_file_integrity", return_value=True), \
            mock.patch.object(manager, "_add_to_manifest"):
        manager._download_worker(task_id, repo_id, filename, model_info)
        assert manager._download_tasks[task_id].status == ModelDownloadStatus.FAILED
        assert "fail" in manager._download_tasks[task_id].error

def test_get_download_status_returns_none_if_missing(manager):
    assert manager.get_download_status("notask") is None
    

def test_add_to_manifest_adds_new_entry(manager):
    repo_id = "org/test"
    filename = "model.gguf"
    model_path = "/tmp/model.gguf"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id=repo_id,
        filename=filename,
        local_path=model_path,
        downloaded_at=datetime.now(),
        file_size=42,
        created_at=datetime.now(),
        downloads=3,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    manager.manifest = {"models": [], "last_updated": None}
    manager._add_to_manifest(repo_id, filename, model_path, model_info)
    assert any(m["repo_id"] == repo_id and m["filename"] == filename for m in manager.manifest["models"])

def test_add_to_manifest_updates_existing_entry(manager):
    repo_id = "org/test"
    filename = "model.gguf"
    model_path = "/tmp/model.gguf"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id=repo_id,
        filename=filename,
        local_path=model_path,
        downloaded_at=datetime.now(),
        file_size=42,
        created_at=datetime.now(),
        downloads=3,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    manager.manifest = {"models": [{
        "repo_id": repo_id,
        "filename": filename,
        "local_path": model_path,
        "downloaded_at": datetime.now().isoformat(),
        "file_size": 10,
        "created_at": datetime.now().isoformat(),
        "downloads": 1,
        "likes": 0,
        "quantization_bit": "Q4",
        "quantization_scheme": "K",
        "quantization_modifier": "default",
        "organization": "org"
    }], "last_updated": None}
    manager._add_to_manifest(repo_id, filename, model_path, model_info)
    found = [m for m in manager.manifest["models"] if m["repo_id"] == repo_id and m["filename"] == filename]
    assert len(found) == 1
    assert found[0]["file_size"] == 42, f"Expected file size to be updated to 42, got {found[0]['file_size']}"

def test_get_download_status_returns_task_info(manager):
    task_id = "task_test"
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="org/model",
        filename="file.gguf",
        local_path="/tmp/file.gguf",
        downloaded_at=datetime.now(),
        file_size=123,
        created_at=datetime.now(),
        downloads=1,
        likes=1,
        quantization_bit="Q4",
        quantization_scheme="K",
        quantization_modifier="default",
    )
    info = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.DOWNLOADING,
        percentage=0.5,
        repo_id="org/model",
        filename="file.gguf",
        created_at=datetime.now(),
        message="Downloading...",
        model_info=model_info,
    )
    manager._download_tasks[task_id] = info
    result = manager.get_download_status(task_id)
    assert result is not None
    assert result["task_id"] == task_id
    assert result["status"] == ModelDownloadStatus.DOWNLOADING
    assert result["repo_id"] == "org/model"
    assert result["filename"] == "file.gguf"

def test_get_download_status_returns_none_for_missing(manager):
    assert manager.get_download_status("missing_task_id") is None




