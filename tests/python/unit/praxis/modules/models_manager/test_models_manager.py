import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
from huggingface_hub.errors import HfHubHTTPError

from ataraxai.praxis.modules.models_manager.models_manager import (
    LlamaCPPModelInfo,
    ModelDownloadInfo,
    ModelDownloadStatus,
    ModelsManager,
)
from ataraxai.praxis.utils.app_directories import AppDirectories


def test_llama_cpp_model_info_is_valid():
    info = LlamaCPPModelInfo(
        organization="test_org",
        repo_id="test/repo",
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        file_size=12345,
        downloads=10,
        likes=5,
    )
    assert info.is_valid()


def test_llama_cpp_model_info_is_invalid_missing_fields():
    info = LlamaCPPModelInfo(
        organization="",
        repo_id="",
        filename="",
        local_path="",
        file_size=-1,
        downloads=0,
        likes=0,
    )
    assert not info.is_valid()


def test_model_download_info_status_validator():
    info = ModelDownloadInfo(
        task_id="abc",
        status=ModelDownloadStatus.STARTING,
        repo_id="repo",
        filename="file.gguf",
    )
    assert info.status == ModelDownloadStatus.STARTING

    with pytest.raises(ValueError):
        ModelDownloadInfo(
            task_id="abc",
            status="not_a_status",
            repo_id="repo",
            filename="file.gguf",
        )


def test_models_manager_load_and_save_manifest(tmp_path: Path):
    manifest_path = tmp_path / "models" / "models.json"
    manifest_path.parent.mkdir()
    manifest_data: Dict[str, Any] = {
        "models": [{"repo_id": "repo", "filename": "file.gguf"}],
        "last_updated": None,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    assert manager.manifest["models"][0]["repo_id"] == "repo"
    manager.manifest["models"].append({"repo_id": "repo2", "filename": "file2.gguf"})
    manager._save_manifest()
    with open(manager.manifest_path) as f:
        saved = json.load(f)
    assert any(m["repo_id"] == "repo2" for m in saved["models"])


def test_get_list_of_models_from_manifest_filters_correctly():
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = Path(tempfile.mkdtemp())
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    manager.manifest = {
        "models": [
            {"repo_id": "repo1", "filename": "file1.gguf", "organization": "org1"},
            {"repo_id": "repo2", "filename": "file2.gguf", "organization": "org2"},
        ],
        "last_updated": None,
    }
    results = manager.get_list_of_models_from_manifest({"repo_id": "repo1"})
    assert len(results) == 1
    assert results[0]["repo_id"] == "repo1"
    results = manager.get_list_of_models_from_manifest({"organization": "org2"})
    assert len(results) == 1
    assert results[0]["organization"] == "org2"
    results = manager.get_list_of_models_from_manifest({"filename": "file"})
    assert len(results) == 2


def test_calculate_sha256(tmp_path: Path):
    file_path = tmp_path / "testfile"
    content = b"hello world"
    file_path.write_bytes(content)
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    expected_hash = hashlib.sha256(content).hexdigest()
    assert manager._calculate_sha256(file_path) == expected_hash


def test_get_expected_sha256_success(monkeypatch):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = Path(tempfile.mkdtemp())
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)

    class DummySibling:
        rfilename = "file.gguf"
        lfs = {"sha256": "abc123"}

    class DummyModelInfo:
        siblings = [DummySibling()]

    monkeypatch.setattr(
        manager.hf_api, "model_info", lambda repo_id, files_metadata: DummyModelInfo()
    )
    assert manager._get_expected_sha256("repo", "file.gguf") == "abc123"


def test_get_expected_sha256_not_found(monkeypatch):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = Path(tempfile.mkdtemp())
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)

    class DummyModelInfo:
        siblings = []

    monkeypatch.setattr(
        manager.hf_api, "model_info", lambda repo_id, files_metadata: DummyModelInfo()
    )
    assert manager._get_expected_sha256("repo", "file.gguf") is None


def test_get_expected_sha256_http_error(monkeypatch):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = Path(tempfile.mkdtemp())
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)

    def raise_error(repo_id, files_metadata):
        raise HfHubHTTPError("error")

    monkeypatch.setattr(manager.hf_api, "model_info", raise_error)
    assert manager._get_expected_sha256("repo", "file.gguf") is None


def test_verify_file_integrity(monkeypatch, tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    file_path = tmp_path / "file.gguf"
    file_path.write_bytes(b"abc")
    monkeypatch.setattr(
        manager,
        "_get_expected_sha256",
        lambda repo_id, filename: hashlib.sha256(b"abc").hexdigest(),
    )
    assert manager._verify_file_integrity(file_path, "repo", "file.gguf")
    monkeypatch.setattr(
        manager, "_get_expected_sha256", lambda repo_id, filename: "wronghash"
    )
    assert not manager._verify_file_integrity(file_path, "repo", "file.gguf")
    monkeypatch.setattr(manager, "_get_expected_sha256", lambda repo_id, filename: None)
    assert not manager._verify_file_integrity(file_path, "repo", "file.gguf")


def test_list_available_files(monkeypatch):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = Path(tempfile.mkdtemp())
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    monkeypatch.setattr(
        manager.hf_api,
        "list_repo_files",
        lambda repo_id: ["a.gguf", "b.bin", "c.txt", "d.safetensors"],
    )
    files = manager.list_available_files("repo")
    assert set(files) == {"a.gguf", "b.bin", "d.safetensors"}


def test_list_available_files_error(monkeypatch):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = Path(tempfile.mkdtemp())
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    monkeypatch.setattr(
        manager.hf_api,
        "list_repo_files",
        lambda repo_id: (_ for _ in ()).throw(Exception("fail")),
    )
    files = manager.list_available_files("repo")
    assert files == []


def test_add_to_manifest_and_list_downloaded_models(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="repo",
        filename="file.gguf",
        local_path=str(tmp_path / "file.gguf"),
        file_size=100,
        downloads=1,
        likes=1,
    )
    manager.manifest = {"models": [], "last_updated": None}
    manager._add_to_manifest(
        "repo", "file.gguf", str(tmp_path / "file.gguf"), model_info
    )
    assert any(
        m["repo_id"] == "repo" and m["filename"] == "file.gguf"
        for m in manager.list_downloaded_models()
    )


def test_get_download_status_and_cancel_download(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    task_id = "task123"
    info = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.STARTING,
        repo_id="repo",
        filename="file.gguf",
    )
    manager._download_tasks[task_id] = info
    status = manager.get_download_status(task_id)
    assert status["task_id"] == task_id
    manager._download_tasks[task_id].status = ModelDownloadStatus.COMPLETED
    assert manager.cancel_download(task_id) is True

    assert task_id in manager._download_tasks
    assert manager._download_tasks[task_id].status == ModelDownloadStatus.CANCELLED
    assert manager._download_tasks[task_id].cancelled_at is not None


def test_cleanup_old_tasks_removes_old(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    old_time = datetime.now().replace(year=2000)
    manager._download_tasks = {
        "t1": ModelDownloadInfo(
            task_id="t1",
            status=ModelDownloadStatus.COMPLETED,
            repo_id="repo",
            filename="file.gguf",
            created_at=old_time,
        ),
        "t2": ModelDownloadInfo(
            task_id="t2",
            status=ModelDownloadStatus.FAILED,
            repo_id="repo",
            filename="file2.gguf",
            created_at=old_time,
        ),
        "t3": ModelDownloadInfo(
            task_id="t3",
            status=ModelDownloadStatus.DOWNLOADING,
            repo_id="repo",
            filename="file3.gguf",
            created_at=datetime.now(),
        ),
    }
    manager.cleanup_old_tasks(max_age_hours=1)
    assert "t3" in manager._download_tasks
    assert "t1" not in manager._download_tasks
    assert "t2" not in manager._download_tasks


@pytest.mark.asyncio
async def test_remove_all_models(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    file_path = tmp_path / "models" / "file.gguf"
    file_path.parent.mkdir(exist_ok=True)
    file_path.write_bytes(b"abc")
    manager.manifest = {
        "models": [
            {"repo_id": "repo", "filename": "file.gguf", "local_path": str(file_path)}
        ],
        "last_updated": None,
    }
    result = await manager.remove_all_models()
    assert result is True
    assert manager.manifest["models"] == []
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_remove_model(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    file_path = tmp_path / "models" / "file.gguf"
    file_path.parent.mkdir(exist_ok=True)
    file_path.write_bytes(b"abc")
    manager.manifest = {
        "models": [
            {"repo_id": "repo", "filename": "file.gguf", "local_path": str(file_path)},
            {"repo_id": "repo2", "filename": "file2.gguf", "local_path": "not_exist"},
        ],
        "last_updated": None,
    }
    result = await manager.remove_model("repo", "file.gguf")
    assert result is True
    assert all(m["repo_id"] != "repo" for m in manager.manifest["models"])
    assert not file_path.exists()


def test_progress_callback_wrapper_updates_task(monkeypatch, tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    task_id = "task1"
    manager._download_tasks[task_id] = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.DOWNLOADING,
        repo_id="repo",
        filename="file.gguf",
    )
    called = {}

    def user_callback(downloaded, total):
        called["downloaded"] = downloaded
        called["total"] = total

    cb = manager._progress_callback_wrapper(task_id, user_callback)
    cb(50, 100)
    assert manager._download_tasks[task_id].downloaded_bytes == 50
    assert manager._download_tasks[task_id].percentage == 0.5
    assert called["downloaded"] == 50
    assert called["total"] == 100


def test_start_download_task_starts_thread(monkeypatch, tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="repo",
        filename="file.gguf",
        local_path=str(tmp_path / "file.gguf"),
        file_size=100,
        downloads=1,
        likes=1,
    )
    task_id = "taskid"
    called = {}

    def dummy_download_worker(*args, **kwargs):
        called["worker"] = True

    monkeypatch.setattr(manager, "_download_worker", dummy_download_worker)
    tid = manager.start_download_task(task_id, model_info)
    assert tid == task_id
    assert background_task_manager.register_task.called


def test_progress_callback_wrapper_calls_user_and_updates(monkeypatch, tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    task_id = "task1"
    manager._download_tasks[task_id] = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.DOWNLOADING,
        repo_id="repo",
        filename="file.gguf",
    )
    called = {}

    def user_callback(downloaded, total):
        called["downloaded"] = downloaded
        called["total"] = total

    cb = manager._progress_callback_wrapper(task_id, user_callback)
    cb(75, 150)
    assert manager._download_tasks[task_id].downloaded_bytes == 75
    assert manager._download_tasks[task_id].percentage == 0.5
    assert called["downloaded"] == 75
    assert called["total"] == 150


def test_add_to_manifest_adds_model(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    manager.manifest = {"models": [], "last_updated": None}
    model_info = LlamaCPPModelInfo(
        organization="org",
        repo_id="repo",
        filename="file.gguf",
        local_path=str(tmp_path / "file.gguf"),
        file_size=100,
        downloads=1,
        likes=1,
    )
    manager._add_to_manifest(
        "repo", "file.gguf", str(tmp_path / "file.gguf"), model_info
    )
    assert any(
        m["repo_id"] == "repo" and m["filename"] == "file.gguf"
        for m in manager.manifest["models"]
    )


def test_list_downloaded_models_returns_models(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    manager.manifest = {
        "models": [
            {"repo_id": "repo", "filename": "file.gguf", "organization": "org"},
            {"repo_id": "repo2", "filename": "file2.gguf", "organization": "org2"},
        ],
        "last_updated": None,
    }
    models = manager.list_downloaded_models()
    assert len(models) == 2
    assert models[0]["repo_id"] == "repo"
    assert models[1]["repo_id"] == "repo2"


def test_get_download_status_returns_status(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    task_id = "taskid"
    info = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.STARTING,
        repo_id="repo",
        filename="file.gguf",
    )
    manager._download_tasks[task_id] = info
    status = manager.get_download_status(task_id)
    assert status["task_id"] == task_id
    assert status["status"] == ModelDownloadStatus.STARTING.value


def test_get_download_status_returns_none_for_missing(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    status = manager.get_download_status("not_exist")
    assert status is None


def test_cancel_download_removes_task(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    task_id = "taskid"
    info = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.COMPLETED,
        repo_id="repo",
        filename="file.gguf",
    )
    manager._download_tasks[task_id] = info
    result = manager.cancel_download(task_id)
    assert result is True

    assert task_id in manager._download_tasks
    assert manager._download_tasks[task_id].status == ModelDownloadStatus.CANCELLED
    assert manager._download_tasks[task_id].cancelled_at is not None


def test_cancel_download_marks_task_as_cancelled(tmp_path: Path):
    """Test that cancel_download marks task as cancelled rather than removing it."""
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    task_id = "taskid"
    info = ModelDownloadInfo(
        task_id=task_id,
        status=ModelDownloadStatus.COMPLETED,
        repo_id="repo",
        filename="file.gguf",
    )
    manager._download_tasks[task_id] = info
    result = manager.cancel_download(task_id)
    assert result is True

    # Task should remain in _download_tasks but be marked as cancelled
    assert task_id in manager._download_tasks
    assert manager._download_tasks[task_id].status == ModelDownloadStatus.CANCELLED
    assert manager._download_tasks[task_id].cancelled_at is not None


# Additional test to verify the cleanup behavior still works for cancelled tasks
def test_cleanup_old_tasks_removes_old_cancelled_tasks(tmp_path: Path):
    """Test that cleanup removes old cancelled tasks."""
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    old_time = datetime.now() - timedelta(hours=2)

    manager._download_tasks = {
        "old_cancelled": ModelDownloadInfo(
            task_id="old_cancelled",
            status=ModelDownloadStatus.CANCELLED,
            repo_id="repo",
            filename="file.gguf",
            created_at=old_time,
            cancelled_at=old_time,
        ),
        "new_cancelled": ModelDownloadInfo(
            task_id="new_cancelled",
            status=ModelDownloadStatus.CANCELLED,
            repo_id="repo",
            filename="file2.gguf",
            created_at=datetime.now(),
            cancelled_at=datetime.now(),
        ),
        "downloading": ModelDownloadInfo(
            task_id="downloading",
            status=ModelDownloadStatus.DOWNLOADING,
            repo_id="repo",
            filename="file3.gguf",
            created_at=datetime.now(),
        ),
    }

    manager.cleanup_old_tasks(max_age_hours=1)

    assert "old_cancelled" not in manager._download_tasks
    assert "new_cancelled" in manager._download_tasks
    assert "downloading" in manager._download_tasks


def test_cancel_download_returns_false_if_not_found(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    result = manager.cancel_download("not_exist")
    assert result is False


def test_cleanup_old_tasks_removes_old_tasks(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    old_time = datetime.now() - timedelta(hours=2)
    manager._download_tasks = {
        "old": ModelDownloadInfo(
            task_id="old",
            status=ModelDownloadStatus.COMPLETED,
            repo_id="repo",
            filename="file.gguf",
            created_at=old_time,
        ),
        "new": ModelDownloadInfo(
            task_id="new",
            status=ModelDownloadStatus.DOWNLOADING,
            repo_id="repo",
            filename="file2.gguf",
            created_at=datetime.now(),
        ),
    }
    manager.cleanup_old_tasks(max_age_hours=1)
    assert "old" not in manager._download_tasks
    assert "new" in manager._download_tasks


@pytest.mark.asyncio
async def test_remove_model_removes_file_and_manifest(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    file_path = tmp_path / "models" / "file.gguf"
    file_path.parent.mkdir(exist_ok=True)
    file_path.write_bytes(b"abc")
    manager.manifest = {
        "models": [
            {"repo_id": "repo", "filename": "file.gguf", "local_path": str(file_path)},
            {"repo_id": "repo2", "filename": "file2.gguf", "local_path": "not_exist"},
        ],
        "last_updated": None,
    }
    result = await manager.remove_model("repo", "file.gguf")
    assert result is True
    assert all(m["repo_id"] != "repo" for m in manager.manifest["models"])
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_remove_all_models_removes_all_files(tmp_path: Path):
    logger = mock.Mock()
    directories = mock.Mock()
    directories.data = tmp_path
    background_task_manager = mock.Mock()
    manager = ModelsManager(directories, logger, background_task_manager)
    file_path1 = tmp_path / "models" / "file1.gguf"
    file_path2 = tmp_path / "models" / "file2.gguf"
    file_path1.parent.mkdir(exist_ok=True)
    file_path1.write_bytes(b"abc")
    file_path2.write_bytes(b"def")
    manager.manifest = {
        "models": [
            {
                "repo_id": "repo1",
                "filename": "file1.gguf",
                "local_path": str(file_path1),
            },
            {
                "repo_id": "repo2",
                "filename": "file2.gguf",
                "local_path": str(file_path2),
            },
        ],
        "last_updated": None,
    }
    result = await manager.remove_all_models()
    assert result is True
    assert manager.manifest["models"] == []
    assert not file_path1.exists()
    assert not file_path2.exists()
