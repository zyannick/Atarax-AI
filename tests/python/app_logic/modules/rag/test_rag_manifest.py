import os
import json
import tempfile
import pytest
from pathlib import Path
from ataraxai.app_logic.modules.rag.rag_manifest import RAGManifest

@pytest.fixture
def temp_manifest_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        yield manifest_path

def test_init_creates_empty_data_if_file_not_exists(temp_manifest_file):
    manifest = RAGManifest(temp_manifest_file)
    assert manifest.data == {}
    assert manifest.path == temp_manifest_file

def test_loads_existing_manifest(temp_manifest_file):
    data = {"file1.txt": {"meta": 1}}
    with open(temp_manifest_file, "w") as f:
        json.dump(data, f)
    manifest = RAGManifest(temp_manifest_file)
    assert manifest.data == data

def test_save_writes_data_to_file(temp_manifest_file):
    manifest = RAGManifest(temp_manifest_file)
    manifest.data = {"foo.txt": {"bar": 2}}
    manifest.save()
    with open(temp_manifest_file) as f:
        loaded = json.load(f)
    assert loaded == {"foo.txt": {"bar": 2}}

def test_add_file_adds_and_saves(temp_manifest_file):
    manifest = RAGManifest(temp_manifest_file)
    manifest.add_file("a.txt", {"size": 123})
    assert "a.txt" in manifest.data
    assert manifest.data["a.txt"] == {"size": 123}
    with open(temp_manifest_file) as f:
        loaded = json.load(f)
    assert "a.txt" in loaded

def test_add_file_with_no_metadata(temp_manifest_file):
    manifest = RAGManifest(temp_manifest_file)
    manifest.add_file("b.txt")
    assert manifest.data["b.txt"] == {}

def test_remove_file_removes_and_saves(temp_manifest_file, capsys):
    manifest = RAGManifest(temp_manifest_file)
    manifest.add_file("c.txt", {"foo": "bar"})
    manifest.remove_file("c.txt")
    assert "c.txt" not in manifest.data
    with open(temp_manifest_file) as f:
        loaded = json.load(f)
    assert "c.txt" not in loaded

def test_remove_file_not_found_prints_message(temp_manifest_file, capsys):
    manifest = RAGManifest(temp_manifest_file)
    manifest.remove_file("notfound.txt")
    captured = capsys.readouterr()
    assert "notfound.txt" in captured.out