from __future__ import annotations
from typing import Optional
from uuid import UUID
from pathlib import Path
from ataraxai.praxis.utils.exceptions import ValidationError



class InputValidator:

    @staticmethod
    def validate_uuid(
        uuid_value: Optional[UUID], param_name: str, version: int = 4
    ) -> None:
        """
        Validates that the provided UUID value is not empty.

        Args:
            uuid_value (Optional[UUID]): The UUID value to validate.
            param_name (str): The name of the parameter being validated, used in the error message.

        Raises:
            ValidationError: If the uuid_value is None or empty.
        """
        
        if isinstance(uuid_value, UUID):
            return
        
        try:
            _ = UUID(uuid_value, version=version)  # type: ignore
        except ValueError:
            raise ValidationError(f"{param_name} is not a valid UUID: {uuid_value}")

    @staticmethod
    def validate_string(string_value: Optional[str], param_name: str) -> None:
        """
        Validates that the provided string is not None, empty, or only whitespace.

        Args:
            string_value (Optional[str]): The string value to validate.
            param_name (str): The name of the parameter being validated, used in the error message.

        Raises:
            ValidationError: If the string_value is None, empty, or contains only whitespace.
        """
        if not string_value or not string_value.strip():
            raise ValidationError(f"{param_name} cannot be empty.")

    @staticmethod
    def validate_path(
        path_value: Optional[str], param_name: str, must_exist: bool = True
    ) -> None:
        """
        Validates a file or directory path.

        Args:
            path_value (Optional[str]): The path to validate.
            param_name (str): The name of the parameter (used in error messages).
            must_exist (bool, optional): If True, the path must exist. Defaults to True.

        Raises:
            ValidationError: If the path is empty or, when must_exist is True, does not exist.
        """
        if not path_value:
            raise ValidationError(f"{param_name} cannot be empty.")

        path = Path(path_value)
        if must_exist and not path.exists():
            raise ValidationError(f"{param_name} path does not exist: {path_value}")

    @staticmethod
    def validate_directory(directory_path: Optional[str], param_name: str) -> None:
        """
        Validates that the provided directory path is a non-empty string and points to an existing directory.

        Args:
            directory_path (Optional[str]): The path to the directory to validate.
            param_name (str): The name of the parameter being validated, used in error messages.

        Raises:
            ValidationError: If the directory_path is empty or does not point to a valid directory.
        """
        if not directory_path:
            raise ValidationError(f"{param_name} cannot be empty.")

        path = Path(directory_path)
        if not path.is_dir():
            raise ValidationError(
                f"{param_name} is not a valid directory: {directory_path}"
            )