from pathlib import Path

def validate_directory_path(path: str) -> bool:
    try:
        resolved = Path(path).resolve()
        return resolved.exists() and resolved.is_dir()
    except Exception:
        return False
    
