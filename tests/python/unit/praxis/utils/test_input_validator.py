import pytest
from uuid import uuid4
from ataraxai.praxis.utils.input_validator import InputValidator
from ataraxai.praxis.utils.exceptions import ValidationError


def test_validate_uuid_valid():
    valid_uuid = str(uuid4())
    InputValidator.validate_uuid(valid_uuid, "test_uuid")


def test_validate_uuid_invalid():
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_uuid("not-a-uuid", "test_uuid")
    assert "test_uuid is not a valid UUID" in str(excinfo.value)


def test_validate_string_valid():
    InputValidator.validate_string("hello", "test_string")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_validate_string_invalid(value):
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_string(value, "test_string")
    assert "test_string cannot be empty." in str(excinfo.value)


def test_validate_path_valid(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    InputValidator.validate_path(str(file_path), "test_path")


def test_validate_path_not_exist():
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_path("/non/existent/path", "test_path")
    assert "test_path path does not exist" in str(excinfo.value)


def test_validate_path_empty():
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_path("", "test_path")
    assert "test_path cannot be empty." in str(excinfo.value)


def test_validate_path_must_exist_false():
    # Should not raise even if path doesn't exist
    InputValidator.validate_path("/non/existent/path", "test_path", must_exist=False)


def test_validate_directory_valid(tmp_path):
    InputValidator.validate_directory(str(tmp_path), "test_dir")


def test_validate_directory_not_exist():
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_directory("/non/existent/dir", "test_dir")
    assert "test_dir is not a valid directory" in str(excinfo.value)


def test_validate_directory_empty():
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_directory("", "test_dir")
    assert "test_dir cannot be empty." in str(excinfo.value)


def test_validate_directory_file_instead_of_dir(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    with pytest.raises(ValidationError) as excinfo:
        InputValidator.validate_directory(str(file_path), "test_dir")
    assert "test_dir is not a valid directory" in str(excinfo.value)
