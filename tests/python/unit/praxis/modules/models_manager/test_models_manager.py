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
from huggingface_hub.errors import HfHubHTTPError

from ataraxai.praxis.modules.models_manager.models_manager import (
    ModelsManager,
    LlamaCPPModelInfo,
    ModelDownloadStatus,
    ModelDownloadInfo,
)


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


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
    manager.manifest = {
        "models": [{"repo_id": "test", "filename": "file.gguf"}],
        "last_updated": None,
    }
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
    with mock.patch.object(
        manager.hf_api, "list_repo_files", return_value=["a.gguf", "b.bin", "c.txt"]
    ):
        files = manager.list_available_files("dummy/repo")
        assert "a.gguf" in files
        assert "b.bin" in files
        assert "c.txt" not in files


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
    assert any(
        m["repo_id"] == repo_id and m["filename"] == filename
        for m in manager.manifest["models"]
    )


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
        called["downloaded"] = downloaded
        called["total"] = total

    cb = manager._progress_callback_wrapper(task_id, user_callback)
    cb(50, 100)
    info = manager._download_tasks[task_id]
    assert info.percentage == 0.5
    assert info.downloaded_bytes == 50
    assert info.file_size == 100
    assert called["downloaded"] == 50
    assert called["total"] == 100


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
    with mock.patch.object(
        manager, "_download_with_progress", return_value=model_path
    ), mock.patch.object(
        manager, "_verify_file_integrity", return_value=True
    ), mock.patch.object(
        manager, "_add_to_manifest"
    ):
        manager._download_worker(task_id, repo_id, filename, model_info)
        assert manager._download_tasks[task_id].status == ModelDownloadStatus.COMPLETED

    manager._download_tasks[task_id].status = ModelDownloadStatus.STARTING
    with mock.patch.object(
        manager, "_download_with_progress", side_effect=Exception("fail")
    ), mock.patch.object(
        manager, "_verify_file_integrity", return_value=True
    ), mock.patch.object(
        manager, "_add_to_manifest"
    ):
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
    assert any(
        m["repo_id"] == repo_id and m["filename"] == filename
        for m in manager.manifest["models"]
    )


def test_get_download_status_returns_none_for_missing(manager):
    assert manager.get_download_status("missing_task_id") is None


def test_get_list_of_models_from_manifest_partial_and_case_insensitive(manager):
    manager.manifest = {
        "models": [
            {"repo_id": "Org/ModelA", "filename": "FileA.gguf", "organization": "Org"},
            {"repo_id": "org/modelb", "filename": "fileB.gguf", "organization": "org"},
            {
                "repo_id": "AnotherOrg/ModelC",
                "filename": "fileC.bin",
                "organization": "AnotherOrg",
            },
            {
                "repo_id": "AtaraxAI/ModelD",
                "filename": "testfile.gguf",
                "organization": "AtaraxAI",
            },
        ],
        "last_updated": datetime.now(),
    }
    results = manager.get_list_of_models_from_manifest({"repo_id": "org/model"})
    assert len(results) == 3
    results = manager.get_list_of_models_from_manifest({"filename": "filea"})
    assert len(results) == 1
    results = manager.get_list_of_models_from_manifest({"organization": "another"})
    assert len(results) == 1
    results = manager.get_list_of_models_from_manifest(
        {"repo_id": "org", "filename": "fileb"}
    )
    assert len(results) == 1
    results = manager.get_list_of_models_from_manifest({"repo_id": "notfound"})
    assert results == []


def test__verify_file_integrity_returns_false_if_no_expected_hash(manager, tmp_path):
    file = tmp_path / "testfile.gguf"
    file.write_bytes(b"dummy")
    with mock.patch.object(manager, "_get_expected_sha256", return_value=None):
        result = manager._verify_file_integrity(file, "repo", "testfile.gguf")
        assert result is False
        assert any(
            "Could not retrieve expected checksum" in msg
            for level, msg in manager.logger.messages
            if level == "error"
        )


def test__verify_file_integrity_returns_true_on_match(manager, tmp_path):
    file = tmp_path / "testfile.gguf"
    file.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    with mock.patch.object(manager, "_get_expected_sha256", return_value=sha):
        result = manager._verify_file_integrity(file, "repo", "testfile.gguf")
        assert result is True
        assert any(
            "Integrity check passed" in msg
            for level, msg in manager.logger.messages
            if level == "info"
        )


def test__verify_file_integrity_returns_false_on_mismatch(manager, tmp_path):
    file = tmp_path / "testfile.gguf"
    file.write_bytes(b"abc")
    wrong_sha = "0" * 64
    with mock.patch.object(manager, "_get_expected_sha256", return_value=wrong_sha):
        result = manager._verify_file_integrity(file, "repo", "testfile.gguf")
        assert result is False
        assert any(
            "Integrity check FAILED" in msg
            for level, msg in manager.logger.messages
            if level == "error"
        )


def test__get_expected_sha256_success(manager):
    class DummySibling:
        def __init__(self, rfilename, sha):
            self.rfilename = rfilename
            self.lfs = {"sha256": sha}

    class DummyModelInfo:
        siblings = [DummySibling("file.gguf", "abc123")]

    with mock.patch.object(manager.hf_api, "model_info", return_value=DummyModelInfo()):
        sha = manager._get_expected_sha256("repo", "file.gguf")
        assert sha == "abc123"


def test__get_expected_sha256_file_not_found(manager):
    class DummySibling:
        def __init__(self, rfilename, sha):
            self.rfilename = rfilename
            self.lfs = {"sha256": sha}

    class DummyModelInfo:
        siblings = [DummySibling("otherfile.gguf", "abc123")]

    with mock.patch.object(manager.hf_api, "model_info", return_value=DummyModelInfo()):
        sha = manager._get_expected_sha256("repo", "file.gguf")
        assert sha is None
        assert any(
            "Could not find file" in msg
            for level, msg in manager.logger.messages
            if level == "error"
        )


def test__get_expected_sha256_http_error(manager):
    with mock.patch.object(
        manager.hf_api, "model_info", side_effect=HfHubHTTPError("fail")
    ):
        sha = manager._get_expected_sha256("repo", "file.gguf")
        assert sha is None
        assert any(
            "Failed to fetch model info" in msg
            for level, msg in manager.logger.messages
            if level == "error"
        )


def test_get_list_of_models_from_manifest_empty(manager):
    manager.manifest = {"models": [], "last_updated": None}
    results = manager.get_list_of_models_from_manifest({"repo_id": "anything"})
    assert results == []


def test_get_list_of_models_from_manifest_no_search_infos(manager):
    manager.manifest = {
        "models": [
            {"repo_id": "a/b", "filename": "file1.gguf", "organization": "a"},
            {"repo_id": "c/d", "filename": "file2.gguf", "organization": "c"},
        ],
        "last_updated": None,
    }
    results = manager.get_list_of_models_from_manifest({})
    assert len(results) == 2


def test_calculate_sha256_empty_file(tmp_path, manager):
    file = tmp_path / "empty.bin"
    file.write_bytes(b"")
    hash_val = manager._calculate_sha256(file)
    expected = hashlib.sha256(b"").hexdigest()
    assert hash_val == expected


def test_save_manifest_handles_exception(manager, tmp_path):
    manager.manifest_path = tmp_path / "readonly.json"
    with mock.patch("builtins.open", side_effect=IOError("fail")):
        manager._save_manifest()
    assert any(
        "Failed to save manifest" in msg
        for level, msg in manager.logger.messages
        if level == "error"
    )


def test_load_manifest_handles_exception(manager, tmp_path):
    manager.manifest_path = tmp_path / "bad.json"
    with mock.patch("builtins.open", side_effect=IOError("fail")):
        manager._load_manifest()
    assert isinstance(manager.manifest, dict)


def test_list_available_files_handles_exception(manager):
    with mock.patch.object(
        manager.hf_api, "list_repo_files", side_effect=Exception("fail")
    ):
        files = manager.list_available_files("repo")
        assert files == []
        assert any(
            "Failed to list files" in msg
            for level, msg in manager.logger.messages
            if level == "error"
        )
