import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from ataraxai.praxis.modules.prompt_engine.prompt_manager import PromptManager

def create_template_file(directory, name, content):
    path = Path(directory) / f"{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def test_load_template_success():
    with TemporaryDirectory() as tmpdir:
        create_template_file(tmpdir, "greeting", "Hello, {name}!")
        pm = PromptManager(Path(tmpdir))
        result = pm.load_template("greeting", name="Alice")
        assert result == "Hello, Alice!"

def test_load_template_missing_placeholder(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        create_template_file(tmpdir, "greeting", "Hello, {name}!")
        pm = PromptManager(Path(tmpdir))
        # Capture print output
        printed = []
        monkeypatch.setattr("builtins.print", lambda msg: printed.append(msg))
        result = pm.load_template("greeting")
        assert result == "Hello, {name}!"
        assert any("Warning: Placeholder" in msg for msg in printed)

def test_load_template_file_not_found():
    with TemporaryDirectory() as tmpdir:
        pm = PromptManager(Path(tmpdir))
        with pytest.raises(FileNotFoundError) as excinfo:
            pm.load_template("nonexistent")
        assert "Prompt template 'nonexistent' not found" in str(excinfo.value)

def test_load_template_directory_not_found():
    with pytest.raises(FileNotFoundError):
        PromptManager(Path("/non/existent/directory"))

def test_load_template_caching():
    with TemporaryDirectory() as tmpdir:
        path = create_template_file(tmpdir, "cached", "Hi, {who}!")
        pm = PromptManager(Path(tmpdir))
        # First call reads from file
        result1 = pm.load_template("cached", who="Bob")
        # Remove file to ensure cache is used
        path.unlink()
        result2 = pm.load_template("cached", who="Bob")
        assert result1 == result2 == "Hi, Bob!"