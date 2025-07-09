import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing_extensions import Union
from ataraxai.app_logic.modules.rag.rag_store import RAGStore

class RAGManifest:
    def __init__(self, manifest_path: Union[str, Path]):
        """
        Initializes the class with the given manifest file path.

        Args:
            manifest_path (str or Path): The path to the manifest file.

        Attributes:
            path (Path): The Path object representing the manifest file location.
            data (Any): The data loaded from the manifest file.
        """
        self.path = Path(manifest_path)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """
        Loads and returns the contents of a JSON file if it exists.

        Returns:
            dict: The parsed JSON data from the file if the file exists, otherwise an empty dictionary.
        """
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def get_all_files(self) -> List[str]:
        """
        Returns a list of all file paths in the manifest.

        Returns:
            list: A list of file paths.
        """
        return list(self.data.keys())

    def save(self):
        """
        Saves the current data to a JSON file at the specified path.

        Opens the file located at `self.path` in write mode and writes the contents
        of `self.data` to it in JSON format with indentation for readability.

        Raises:
            IOError: If the file cannot be opened or written to.
            TypeError: If `self.data` contains non-serializable objects.
        """
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_file(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Adds a file and its associated metadata to the data store.

        Args:
            file_path (str): The path to the file to be added.
            metadata (dict, optional): Additional metadata to associate with the file. Defaults to an empty dictionary if not provided.

        Side Effects:
            Updates the internal data store with the new file and metadata, and persists the changes by calling self.save().
        """
        if not metadata:
            metadata = {}
        self.data[str(file_path)] = metadata
        self.save()

    def is_file_in_manifest(self, file_path: Union[str, Path]) -> bool:
        """
        Checks if a file is in the manifest.

        Args:
            file_path (str or Path): The path to the file to check.

        Returns:
            bool: True if the file is in the manifest, False otherwise.
        """
        return str(file_path) in self.data
    
    def is_valid(self, rag_store: RAGStore) -> bool:
        """
        Validates that all chunk IDs listed in the manifest exist in the provided RAGStore.

        Args:
            rag_store (RAGStore): The RAGStore instance to validate chunk IDs against.

        Returns:
            bool: True if all chunk IDs in the manifest are present in the RAGStore, or if the manifest is empty; False otherwise.

        Notes:
            - If the manifest contains no data or no chunk IDs, the method returns True.
            - Prints a message if the manifest contains no chunk IDs to validate.
        """
        if not self.data:
            return True #

        all_manifest_chunk_ids = set() # type: ignore
        for file_info in self.data.values():
            chunk_ids = file_info.get("chunk_ids")
            if isinstance(chunk_ids, list):
                all_manifest_chunk_ids.update(chunk_ids) # type: ignore

        if not all_manifest_chunk_ids:
            print("Manifest contains no chunk IDs to validate.")
            return True

        retrieved_chunks = rag_store.collection.get(
            ids=list(all_manifest_chunk_ids)
        )
        retrieved_ids = set(retrieved_chunks.get("ids", []))

        if all_manifest_chunk_ids == retrieved_ids:
            return True
        else:
            # missing_in_store = all_manifest_chunk_ids - retrieved_ids # type: ignore
            # extra_in_store = retrieved_ids - all_manifest_chunk_ids
            return False
        
    def clear(self):
        self.data = {}
        self.save()

    def remove_file(self, file_path: Union[str, Path]):
        """
        Remove a file entry from the manifest.

        Args:
            file_path (str): The path of the file to remove from the manifest.

        Returns:
            None

        Side Effects:
            - Removes the specified file entry from the internal data structure if it exists.
            - Saves the updated manifest if the file was found and removed.
            - Prints a message if the file was not found in the manifest.
        """
        if str(file_path) in self.data:
            del self.data[str(file_path)]
            self.save()
        else:
            print(f"File {file_path} not found in manifest.")
