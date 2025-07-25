import pytest
from unittest import mock
from pathlib import Path
import json
import threading
from ataraxai.praxis.modules.models_manager.models_manager import ModelsManager, ModelDownloadStatus
import hashlib
from datetime import datetime, timedelta


@pytest.fixture
def mock_directories(tmp_path):
    class MockDirs:
        data = tmp_path

    return MockDirs()


@pytest.fixture
def mock_logger():
    return mock.Mock()


@pytest.fixture
def model_manager(mock_directories, mock_logger):
    with mock.patch("ataraxai.praxis.modules.models_manager.model_manager.HfApi"):
        return ModelsManager(mock_directories, mock_logger)


def test_init_creates_models_dir_and_manifest(model_manager, mock_directories):
    models_dir = mock_directories.data / "models"
    assert models_dir.exists()
    assert model_manager.models_dir == models_dir
    assert model_manager.manifest_path == models_dir / "models.json"


def test_load_manifest_reads_existing_manifest(model_manager):
    manifest_data = {
        "models": [{"repo_id": "foo", "filename": "bar"}],
        "last_updated": "now",
    }
    with open(model_manager.manifest_path, "w") as f:
        json.dump(manifest_data, f)
    model_manager._load_manifest()
    assert model_manager.manifest == manifest_data


def test_load_manifest_initializes_if_missing(model_manager):
    if model_manager.manifest_path.exists():
        model_manager.manifest_path.unlink()
    model_manager._load_manifest()
    assert model_manager.manifest == {"models": [], "last_updated": None}


def test_save_manifest_writes_file(model_manager):
    model_manager.manifest = {"models": [], "last_updated": None}
    model_manager._save_manifest()
    with open(model_manager.manifest_path) as f:
        data = json.load(f)
    assert "last_updated" in data


def test_calculate_sha256(tmp_path, model_manager):
    file = tmp_path / "test.bin"
    file.write_bytes(b"hello world")
    hash_val = model_manager._calculate_sha256(file)
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert hash_val == expected


def test_get_expected_sha256_success(model_manager):
    mock_hf_api = mock.Mock()
    sibling = mock.Mock()
    sibling.rfilename = "file.gguf"
    sibling.lfs = {"sha256": "abc123"}
    model_info = mock.Mock()
    model_info.siblings = [sibling]
    model_manager.hf_api = mock_hf_api
    mock_hf_api.model_info.return_value = model_info
    result = model_manager._get_expected_sha256("repo", "file.gguf")
    assert result == "abc123"


def test_get_expected_sha256_not_found(model_manager):
    mock_hf_api = mock.Mock()
    model_info = mock.Mock()
    model_info.siblings = []
    model_manager.hf_api = mock_hf_api
    mock_hf_api.model_info.return_value = model_info
    result = model_manager._get_expected_sha256("repo", "file.gguf")
    assert result is None


def test_verify_file_integrity_pass(model_manager, tmp_path):
    file = tmp_path / "file.gguf"
    file.write_bytes(b"abc")
    model_manager._get_expected_sha256 = mock.Mock(
        return_value="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    model_manager._calculate_sha256 = mock.Mock(
        return_value="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert model_manager._verify_file_integrity(file, "repo", "file.gguf") is True


def test_verify_file_integrity_fail(model_manager, tmp_path):
    file = tmp_path / "file.gguf"
    file.write_bytes(b"abc")
    model_manager._get_expected_sha256 = mock.Mock(return_value="wronghash")
    model_manager._calculate_sha256 = mock.Mock(return_value="anotherhash")
    assert model_manager._verify_file_integrity(file, "repo", "file.gguf") is False


def test_search_models_filters_and_returns_list(model_manager):
    mock_hf_api = mock.Mock()
    model = mock.Mock()
    model.id = "repo"
    model.__dict__ = {"id": "repo"}
    mock_hf_api.list_models.return_value = [model]
    mock_hf_api.list_repo_files.return_value = ["file1.gguf", "file2.bin"]
    model_manager.hf_api = mock_hf_api
    result = model_manager.search_models("test", limit=1)
    assert isinstance(result, list)
    assert result[0]["gguf_files"] == ["file1.gguf"]


def test_list_available_files_returns_filtered(model_manager):
    mock_hf_api = mock.Mock()
    mock_hf_api.list_repo_files.return_value = [
        "a.gguf",
        "b.bin",
        "c.safetensors",
        "d.pt",
        "e.pth",
        "f.txt",
    ]
    model_manager.hf_api = mock_hf_api
    files = model_manager.list_available_files("repo")
    assert set(files) == {"a.gguf", "b.bin", "c.safetensors", "d.pt", "e.pth"}


def test_add_to_manifest_adds_and_updates(model_manager, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"abc")
    repo_id = "repo"
    filename = "model.gguf"
    model_manager.manifest = {"models": [], "last_updated": None}
    model_manager._add_to_manifest(repo_id, filename, str(model_path))
    assert any(
        m["repo_id"] == repo_id and m["filename"] == filename
        for m in model_manager.manifest["models"]
    )
    # Update
    model_manager._add_to_manifest(repo_id, filename, str(model_path))
    assert (
        sum(
            m["repo_id"] == repo_id and m["filename"] == filename
            for m in model_manager.manifest["models"]
        )
        == 1
    )


def test_list_downloaded_models_returns_manifest(model_manager):
    model_manager.manifest = {"models": [{"repo_id": "repo", "filename": "file"}]}
    models = model_manager.list_downloaded_models()
    assert models == [{"repo_id": "repo", "filename": "file"}]


def test_remove_model_removes_and_deletes_file(model_manager, tmp_path):
    file = tmp_path / "model.gguf"
    file.write_bytes(b"abc")
    repo_id = "repo"
    filename = "model.gguf"
    model_manager.manifest = {
        "models": [{"repo_id": repo_id, "filename": filename, "local_path": str(file)}]
    }
    # Patch Path.exists and Path.unlink
    with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
        "pathlib.Path.unlink"
    ) as mock_unlink:
        result = model_manager.remove_model(repo_id, filename)
        assert result is True
        mock_unlink.assert_called_once()
    assert not any(
        m["repo_id"] == repo_id and m["filename"] == filename
        for m in model_manager.manifest["models"]
    )


def test_cancel_download_sets_status(model_manager):
    task_id = "task1"
    model_manager._download_tasks[task_id] = {"status": None}
    result = model_manager.cancel_download(task_id)
    assert result is True
    assert model_manager._download_tasks[task_id]["status"] == ModelDownloadStatus.CANCELLED


def test_get_download_status_returns_copy(model_manager):
    task_id = "task1"
    model_manager._download_tasks[task_id] = {"status": ModelDownloadStatus.COMPLETED}
    status = model_manager.get_download_status(task_id)
    assert status == {"status": ModelDownloadStatus.COMPLETED}


def test_cleanup_old_tasks_removes_old(model_manager):
    now = datetime.now()
    old_time = (now - timedelta(hours=25)).isoformat()
    model_manager._download_tasks = {
        "t1": {"created_at": old_time, "status": ModelDownloadStatus.COMPLETED},
        "t2": {"created_at": now.isoformat(), "status": ModelDownloadStatus.DOWNLOADING},
    }
    model_manager.cleanup_old_tasks(max_age_hours=24)
    assert "t1" not in model_manager._download_tasks
    assert "t2" in model_manager._download_tasks
