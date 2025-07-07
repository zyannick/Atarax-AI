import hashlib
from pathlib import Path
from typing import Dict, Any


def get_file_hash(file_path: Path) -> str | None:
    """
    Calculates the SHA-256 hash of a file.

    Args:
        file_path (Path): The path to the file to be hashed.

    Returns:
        str | None: The hexadecimal SHA-256 hash of the file as a string,
        or None if an IOError occurs during file reading.

    Raises:
        None: All exceptions are handled internally.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"Error hashing file {file_path}: {e}")
        return None

def set_base_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Generates and returns a dictionary containing base metadata for a given file.
    Args:
        file_path (Path): The path to the file for which metadata is to be extracted.
    Returns:
        dict: A dictionary with the following keys:
            - "original_filename": The name of the file.
            - "file_path": The string representation of the file path.
            - "file_size": The size of the file in bytes (0 if file does not exist).
            - "file_hash": The hash of the file contents (None if file does not exist).
            - "file_timestamp": The last modification time of the file as an integer timestamp (0 if file does not exist).
    """

    file_size = file_path.stat().st_size if file_path.exists() else 0
    file_hash = get_file_hash(file_path) if file_path.exists() else None
    file_timestamp = int(file_path.stat().st_mtime) if file_path.exists() else 0
    
    return {
        "original_filename": file_path.name,
        "file_path": str(file_path),
        "file_size": file_size,
        "file_hash": file_hash,
        "file_timestamp": file_timestamp,
    }