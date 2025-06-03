import hashlib
from pathlib import Path



def get_file_hash(file_path: Path) -> str | None:
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"Error hashing file {file_path}: {e}")
        return None
    
def set_base_metadata(file_path: Path) -> dict:
    """
    Sets base metadata for a file, including its name, path, size, hash, and timestamp.
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