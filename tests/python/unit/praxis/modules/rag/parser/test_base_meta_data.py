import pytest
import tempfile
import os
from pathlib import Path
from ataraxai.praxis.modules.rag.parser.base_meta_data import get_file_hash
import hashlib

def test_get_file_hash_returns_correct_hash_for_nonempty_file():
    content = b"hello world"
    expected_hash = hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = get_file_hash(tmp_path)
        assert result == expected_hash
    finally:
        os.remove(tmp_path)

def test_get_file_hash_returns_correct_hash_for_empty_file():
    expected_hash = hashlib.sha256(b"").hexdigest()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = get_file_hash(tmp_path)
        assert result == expected_hash
    finally:
        os.remove(tmp_path)

def test_get_file_hash_returns_none_for_missing_file():
    missing_path = Path("this_file_does_not_exist_123456789.txt")
    result = get_file_hash(missing_path)
    assert result is None

def test_get_file_hash_handles_permission_error(monkeypatch):
    def raise_permission_error(*args, **kwargs):
        raise IOError("Permission denied")
    monkeypatch.setattr("builtins.open", raise_permission_error)
    result = get_file_hash(Path("dummy.txt"))
    assert result is None