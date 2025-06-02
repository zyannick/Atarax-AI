import json
from pathlib import Path


class RAGManifest:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_file(self, file_path, metadata=None):
        if not metadata:
            metadata = {}
        self.data[file_path] = metadata
        self.save()

    def remove_file(self, file_path):
        if file_path in self.data:
            del self.data[file_path]
            self.save()
        else:
            print(f"File {file_path} not found in manifest.")
