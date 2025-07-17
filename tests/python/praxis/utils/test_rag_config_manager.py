import pytest
import yaml
from pathlib import Path
from unittest import mock
from ataraxai.praxis.utils.rag_config_manager import RAGConfigManager, RAG_CONFIG_FILENAME


class DummyRAGConfig:
    def __init__(self, foo="bar"):
        self.foo = foo

    def model_dump(self):
        return {"foo": self.foo}

    def __eq__(self, other):
        return isinstance(other, DummyRAGConfig) and self.foo == other.foo

@pytest.fixture(autouse=True)
def patch_rag_config(monkeypatch):
    import ataraxai.praxis.utils.rag_config_manager as rag_config_manager_mod
    monkeypatch.setattr(rag_config_manager_mod, "RAGConfig", DummyRAGConfig)
    yield

@pytest.fixture
def tmp_config_dir(tmp_path):
    return tmp_path

def test_initializes_default_config(tmp_config_dir):
    manager = RAGConfigManager(tmp_config_dir)
    assert manager.get_config() == DummyRAGConfig()
    config_file = tmp_config_dir / RAG_CONFIG_FILENAME
    assert config_file.exists()
    with open(config_file) as f:
        data = yaml.safe_load(f)
    assert data == {"foo": "bar"}

def test_loads_existing_config(tmp_config_dir):
    config_file = tmp_config_dir / RAG_CONFIG_FILENAME
    with open(config_file, "w") as f:
        yaml.dump({"foo": "baz"}, f)
    manager = RAGConfigManager(tmp_config_dir)
    assert manager.get_config() == DummyRAGConfig(foo="baz")

def test_update_config_saves_to_disk(tmp_config_dir):
    manager = RAGConfigManager(tmp_config_dir)
    new_config = DummyRAGConfig(foo="updated")
    manager.update_config(new_config)
    assert manager.get_config() == new_config
    config_file = tmp_config_dir / RAG_CONFIG_FILENAME
    with open(config_file) as f:
        data = yaml.safe_load(f)
    assert data == {"foo": "updated"}

def test_reload_reads_from_disk(tmp_config_dir):
    manager = RAGConfigManager(tmp_config_dir)
    config_file = tmp_config_dir / RAG_CONFIG_FILENAME
    # Change file directly
    with open(config_file, "w") as f:
        yaml.dump({"foo": "reloaded"}, f)
    manager.reload()
    assert manager.get_config() == DummyRAGConfig(foo="reloaded")

def test_load_invalid_yaml_prints_error(tmp_config_dir, capsys):
    config_file = tmp_config_dir / RAG_CONFIG_FILENAME
    with open(config_file, "w") as f:
        f.write("invalid: [unclosed")
    manager = RAGConfigManager(tmp_config_dir)
    captured = capsys.readouterr()
    assert "[ERROR] Failed to load YAML config" in captured.out
    assert manager.get_config() == DummyRAGConfig()