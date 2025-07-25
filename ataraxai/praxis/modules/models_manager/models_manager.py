import os
import random
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import hashlib
import threading
import json
import time
from typing import Any, List, Dict, Optional, Callable
from pathlib import Path
import requests
import tqdm
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from enum import Enum, auto
import ulid
from datetime import datetime
from huggingface_hub import hf_hub_url
from ataraxai.praxis.utils.ataraxai_settings import AtaraxAISettings
import logging
import re
from pydantic import BaseModel, Field, field_validator


class ModelDownloadStatus(Enum):
    STARTING = auto()
    DOWNLOADING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class ModelInfo(BaseModel):
    organization: str
    repo_id: str
    filename: str
    local_path: str
    downloaded_at: str
    file_size: int
    quantization_bit: str = "default"
    quantization_scheme: str = "default"
    quantization_modifier: str = "default"
    created_at: datetime
    downloads: int
    likes: int

    class Config:
        from_attributes = True


class ModelDownloadInfo(BaseModel):
    task_id: str
    status: ModelDownloadStatus
    percentage: float = 0.0
    repo_id: str
    filename: str
    message: str = "Download task started."
    created_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    file_size: Optional[int] = None
    downloaded_bytes: int = 0
    model_info: Optional[ModelInfo] = None
    model_path : Optional[str] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    cancelled_at : Optional[datetime] = None

    @field_validator("status")
    def validate_status(cls, v):
        if not isinstance(v, ModelDownloadStatus):
            raise ValueError("Invalid status type")
        return v

class ModelsManager:
    def __init__(self, directories: AppDirectories, logger: logging.Logger):
        """
        Initializes the ModelManager instance.

        Args:
            directories (AppDirectories): An object containing application directory paths.
            logger (logging.Logger): Logger instance for logging messages.

        Attributes:
            models_dir (Path): Directory path for storing models.
            manifest_path (Path): Path to the models manifest file (models.json).
            hf_api (HfApi): Instance of Hugging Face API for model operations.
            _download_tasks (Dict[str, Any]): Dictionary to track ongoing download tasks.
            _lock (threading.Lock): Thread lock for synchronizing access to download tasks.

        Side Effects:
            Creates the models directory if it does not exist.
            Loads the models manifest from disk.
        """
        self.directories = directories
        self.logger = logger
        self.models_dir = self.directories.data / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.manifest_path = self.models_dir / "models.json"
        self.hf_api = HfApi()
        self._download_tasks: Dict[str, ModelDownloadInfo] = {}
        self._lock = threading.Lock()
        self._load_manifest()

    def _download_with_progress(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Downloads a file from the Hugging Face Hub with a progress bar and optional callback.

        Args:
            repo_id (str): The repository ID on Hugging Face Hub.
            filename (str): The name of the file to download.
            local_dir (str, optional): Directory to save the downloaded file. If None, saves to current directory.
            callback (callable, optional): A function to call with (downloaded_size, total_size) after each chunk.

        Returns:
            str: The local path to the downloaded file.
        """
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
        else:
            local_path = filename
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0
        with open(local_path, "wb") as f:
            with tqdm.tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {filename}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        downloaded_size += len(chunk)
                        if callback:
                            callback(downloaded_size, total_size)

        return local_path

    def _load_manifest(self):
        """
        Loads the model manifest from the specified manifest file path.

        If the manifest file exists, it reads and parses the JSON content into the `self.manifest` attribute.
        If the file does not exist or an error occurs during loading, initializes `self.manifest` with default values.

        Handles exceptions by logging the error and setting a default manifest structure.

        Returns:
            None
        """
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, "r") as f:
                    self.manifest = json.load(f)
            else:
                self.manifest = {"models": [], "last_updated": None}
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            self.manifest = {"models": [], "last_updated": None}

    def _save_manifest(self):
        """
        Saves the current state of the manifest to a JSON file.

        Updates the 'last_updated' field in the manifest with the current timestamp,
        then writes the manifest dictionary to the file specified by 'manifest_path'.
        Logs an error message if the save operation fails.

        Raises:
            Logs any exceptions encountered during the save process.
        """
        try:
            self.manifest["last_updated"] = datetime.now().isoformat()
            with open(self.manifest_path, "w") as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")

    def _calculate_sha256(self, file_path: Path) -> str:
        """
        Calculates the SHA-256 hash of the specified file.

        Args:
            file_path (Path): The path to the file whose SHA-256 hash is to be calculated.

        Returns:
            str: The hexadecimal representation of the SHA-256 hash of the file.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_expected_sha256(self, repo_id: str, filename: str) -> Optional[str]:
        """
        Retrieves the expected SHA-256 checksum for a specific file in a Hugging Face model repository.

        Args:
            repo_id (str): The identifier of the Hugging Face model repository.
            filename (str): The name of the file for which to retrieve the SHA-256 checksum.

        Returns:
            Optional[str]: The SHA-256 checksum of the specified file if available, otherwise None.

        Logs:
            - A warning if the file is not found in the repository.
            - An error if fetching model information fails due to an HTTP error.
        """
        try:
            model_info = self.hf_api.model_info(repo_id=repo_id, files_metadata=True)
            for sibling in model_info.siblings or []:
                if sibling.rfilename == filename:
                    if sibling.lfs:
                        return sibling.lfs.get("sha256")
            self.logger.warning(
                f"Could not find file '{filename}' in repo '{repo_id}' to get checksum."
            )
            return None
        except HfHubHTTPError as e:
            self.logger.error(f"Failed to fetch model info for {repo_id}: {e}")
            return None

    def _verify_file_integrity(
        self, file_path: Path, repo_id: str, filename: str
    ) -> bool:
        """
        Verifies the integrity of a local file by comparing its SHA256 hash with the expected hash.

        Args:
            file_path (Path): The path to the local file to verify.
            repo_id (str): The repository identifier used to retrieve the expected hash.
            filename (str): The name of the file being verified.

        Returns:
            bool: True if the file's integrity is verified or if the expected hash cannot be retrieved; False otherwise.

        Logs:
            - Info message when starting and passing integrity check.
            - Warning if expected checksum cannot be retrieved.
            - Error messages if integrity check fails, including expected and calculated SHA256 hashes.
        """
        self.logger.info(f"Verifying integrity of {filename}...")
        expected_hash = self._get_expected_sha256(repo_id, filename)
        if not expected_hash:
            self.logger.warning(
                f"Could not retrieve expected checksum for {filename}. Skipping verification."
            )
            return True

        local_hash = self._calculate_sha256(file_path)

        if local_hash == expected_hash:
            self.logger.info(f"Integrity check passed for {filename}.")
            return True
        else:
            self.logger.error(f"Integrity check FAILED for {filename}.")
            self.logger.error(f"  - Expected SHA256: {expected_hash}")
            self.logger.error(f"  - Calculated SHA256: {local_hash}")
            return False

    def search_models(
        self, query: str, limit: int = 50, filter_tags: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """
        Searches for models matching the given query and optional filter tags, returning a list of model dictionaries.

        Args:
            query (str): The search query string to filter models.
            limit (int, optional): Maximum number of models to return. Defaults to 50.
            filter_tags (Optional[List[str]], optional): Additional tags to filter models. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries, each representing a model and its associated '.gguf' files.

        Notes:
            - Models are filtered to include those with the 'gguf' tag and any additional tags provided.
            - Results are sorted by download count in descending order.
            - If listing '.gguf' files for a model fails, an empty list is assigned to 'gguf_files'.
            - Errors during the search process are logged and an empty list is returned.
        """
        try:
            filters = ["gguf"]
            if filter_tags:
                filters.extend(filter_tags)

            models = self.hf_api.list_models(
                search=query,
                filter=filters,
                limit=limit,
                sort="downloads",
                direction=-1,
            )

            result = []
            for model in models:
                model_dict = model.__dict__.copy()
                try:
                    files = self.hf_api.list_repo_files(model.id)
                    gguf_files = [f for f in files if f.endswith(".gguf")]
                    model_dict["gguf_files"] = gguf_files
                except Exception as e:
                    self.logger.warning(
                        f"Failed to list gguf files for {model.id}: {e}"
                    )
                    model_dict["gguf_files"] = []

                for gguf_file in model_dict["gguf_files"]:
                    organization = model.id.split("/")[0]
                    match = re.search(r"Q(\d+)_([A-Z])(?:_([A-Z]))?", gguf_file)
                    if match:
                        bits = str(match.group(1))
                        scheme = match.group(2)
                        modifier = match.group(3) or None
                    model_info = ModelInfo(
                        organization=organization,
                        repo_id=model.id,
                        filename=gguf_file,
                        local_path=str(self.models_dir / model.id / gguf_file),
                        downloaded_at=datetime.now().isoformat(),
                        file_size=0,  # Will be set after download
                        created_at=model_dict.get(
                            "created_at", datetime.now().isoformat()
                        ),
                        downloads=model_dict.get("downloads", 0),
                        likes=model_dict.get("likes", 0),
                        quantization_bit=f"Q{bits}" if bits else "default",
                        quantization_scheme=scheme if scheme else "default",
                        quantization_modifier=modifier if modifier else "default",
                    )

                    result.append(model_info)

                # result.append(model_dict)

            return result

        except Exception as e:
            self.logger.error(f"Failed to search models: {e}")
            return []

    def list_available_files(self, repo_id: str) -> List[str]:
        """
        Lists available model files in a Hugging Face repository that match specific file extensions.

        Args:
            repo_id (str): The identifier of the Hugging Face repository.

        Returns:
            List[str]: A list of filenames in the repository that end with one of the following extensions:
                '.gguf', '.bin', '.safetensors', '.pt', or '.pth'.
                Returns an empty list if an error occurs during file listing.

        Logs:
            Errors encountered during file listing are logged.
        """
        try:
            files = self.hf_api.list_repo_files(repo_id)
            return [
                f
                for f in files
                if f.endswith((".gguf", ".bin", ".safetensors", ".pt", ".pth"))
            ]
        except Exception as e:
            self.logger.error(f"Failed to list files for {repo_id}: {e}")
            return []

    def start_download_task(
        self, task_id, model_info: ModelInfo, progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Starts a background task to download a file from the specified repository.

        Args:
            repo_id (str): The identifier of the repository to download from.
            filename (str): The name of the file to download.
            progress_callback (Optional[Callable], optional): A callback function to report download progress. Defaults to None.

        Returns:
            str: A unique task ID representing the download task.
        """


        repo_id = model_info.repo_id
        filename = model_info.filename

        with self._lock:
            self._download_tasks[task_id] = ModelDownloadInfo(
                task_id=task_id,
                status=ModelDownloadStatus.STARTING,
                percentage=0.0,
                repo_id=repo_id,
                filename=filename,
                created_at=datetime.now(),
                message="Download task started.",
                model_info=model_info,
            )

        thread = threading.Thread(
            target=self._download_worker,
            args=(task_id, repo_id, filename, model_info, progress_callback),
            daemon=True,
        )
        thread.start()
        return task_id

    def _progress_callback_wrapper(
        self, task_id: str, user_callback: Optional[Callable] = None
    ):
        """
        Wraps a progress callback function to update internal download task progress and optionally invoke a user-provided callback.

        Args:
            task_id (str): Unique identifier for the download task.
            user_callback (Optional[Callable], optional): A user-defined callback function that takes two arguments (downloaded, total).

        Returns:
            Callable: A callback function that updates the internal progress state and calls the user-provided callback if given.
        """

        def callback(downloaded: int, total: int):
            progress = downloaded / total if total > 0 else 0.0
            with self._lock:
                if task_id in self._download_tasks:
                    self._download_tasks[task_id].percentage = progress
                    self._download_tasks[task_id].downloaded_bytes = downloaded
                    self._download_tasks[task_id].file_size = total

            if user_callback:
                user_callback(downloaded, total)

        return callback

    def _download_worker(
        self,
        task_id: str,
        repo_id: str,
        filename: str,
        model_info: ModelInfo,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Handles the downloading of a model file in a worker thread.

        Args:
            task_id (str): Unique identifier for the download task.
            repo_id (str): Identifier of the model repository.
            filename (str): Name of the file to download.
            progress_callback (Optional[Callable], optional): Callback function to report download progress. Defaults to None.

        Side Effects:
            - Updates the download task status and progress in the internal tracking dictionary.
            - Logs download events and errors.
            - Adds the downloaded model to the manifest upon successful completion.
            - Verifies file integrity after download.

        Exceptions:
            - Catches all exceptions, logs the error, and updates the task status to FAILED.
        """
        try:
            with self._lock:
                self._download_tasks[task_id].status = ModelDownloadStatus.DOWNLOADING

            self.logger.info(f"Starting download: {repo_id}/{filename}")

            callback = self._progress_callback_wrapper(task_id, progress_callback)

            model_path = self._download_with_progress(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                callback=callback,
            )

            if self._verify_file_integrity(Path(model_path), repo_id, filename):
                self.logger.info(f"File integrity verified for {filename}")

            self._add_to_manifest(repo_id, filename, model_path, model_info)

            with self._lock:
                self._download_tasks[task_id].status = ModelDownloadStatus.COMPLETED
                self._download_tasks[task_id].percentage = 1.0
                self._download_tasks[task_id].model_path = str(model_path)
                self._download_tasks[task_id].completed_at = datetime.now()

            self.logger.info(f"Download completed: {repo_id}/{filename}")

        except Exception as e:
            self.logger.error(f"Download failed for {repo_id}/{filename}: {e}")
            with self._lock:
                self._download_tasks[task_id].status = ModelDownloadStatus.FAILED
                self._download_tasks[task_id].error = str(e)
                self._download_tasks[task_id].failed_at = datetime.now()

    def _add_to_manifest(
        self, repo_id: str, filename: str, model_path: str, model_info: ModelInfo
    ):
        """
        Adds or updates model information in the manifest.

        If a model with the given `repo_id` and `filename` already exists in the manifest,
        its information is updated. Otherwise, a new entry is appended.

        Args:
            repo_id (str): The repository identifier for the model.
            filename (str): The filename of the model.
            model_path (str): The local filesystem path to the model file.

        The model information includes repository ID, filename, local path, download timestamp,
        and file size. The manifest is saved after modification.
        """
        model_info.file_size = (
            Path(model_path).stat().st_size if Path(model_path).exists() else 0
        )
        model_info.downloaded_at = datetime.now().isoformat()

        for i, existing in enumerate(self.manifest["models"]):
            if existing["repo_id"] == repo_id and existing["filename"] == filename:
                self.manifest["models"][i] = model_info.model_dump()
                self._save_manifest()
                return

        self.manifest["models"].append(model_info.model_dump())
        self._save_manifest()

    def get_download_status(self, task_id: str) -> Optional[Dict]:
        """
        Retrieves the download status for a given task ID.

        Args:
            task_id (str): The unique identifier of the download task.

        Returns:
            Optional[Dict]: A copy of the download status dictionary for the specified task ID,
                            or None if the task ID does not exist.
        """
        with self._lock:
            model_download_info = self._download_tasks.get(task_id)
            if model_download_info:
                return model_download_info.model_dump()
            else:
                self.logger.warning(f"Download task {task_id} not found.")
                return None

    def cancel_download(self, task_id: str) -> bool:
        """
        Cancels an ongoing model download task by its task ID.

        Args:
            task_id (str): The unique identifier of the download task to cancel.

        Returns:
            bool: True if the task was found and cancelled; False otherwise.
        """
        with self._lock:
            if task_id in self._download_tasks:
                self._download_tasks[task_id].status = ModelDownloadStatus.CANCELLED
                self._download_tasks[task_id].cancelled_at = datetime.now()
                return True
        return False

    def list_downloaded_models(self) -> List[Dict]:
        """
        Returns a list of downloaded models from the manifest.

        Each model is represented as a dictionary containing its metadata.
        If no models are found, returns an empty list.

        Returns:
            List[Dict]: A list of dictionaries, each representing a downloaded model.
        """
        return self.manifest.get("models", [])

    def remove_model(self, repo_id: str, filename: str) -> bool:
        """
        Removes a model from the manifest and deletes its local file if it exists.

        Args:
            repo_id (str): The repository ID of the model to remove.
            filename (str): The filename of the model to remove.

        Returns:
            bool: True if the model was successfully removed, False otherwise.

        Side Effects:
            - Deletes the model file from the local filesystem if present.
            - Updates and saves the manifest.
            - Logs information and errors related to the removal process.
        """
        try:
            for i, model in enumerate(self.manifest["models"]):
                if model["repo_id"] == repo_id and model["filename"] == filename:
                    if model.get("local_path") and Path(model["local_path"]).exists():
                        Path(model["local_path"]).unlink()

                    del self.manifest["models"][i]
                    self._save_manifest()
                    self.logger.info(f"Removed model: {repo_id}/{filename}")
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Failed to remove model {repo_id}/{filename}: {e}")
            return False

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        Removes old download tasks from the internal task list that have exceeded the specified age limit.

        Args:
            max_age_hours (int, optional): The maximum age (in hours) a task can remain before being removed.
                Defaults to 24.

        The method checks each task's creation time and status. Tasks older than `max_age_hours` and with a status of
        COMPLETED, FAILED, or CANCELLED are deleted from the internal tracking dictionary.
        """
        current_time = datetime.now()
        with self._lock:
            to_remove = []
            for task_id, task_data in self._download_tasks.items():
                created_at = task_data.created_at
                age_hours = (current_time - created_at).total_seconds() / 3600

                if age_hours > max_age_hours and task_data.status in [
                    ModelDownloadStatus.COMPLETED,
                    ModelDownloadStatus.FAILED,
                    ModelDownloadStatus.CANCELLED,
                ]:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._download_tasks[task_id]


if __name__ == "__main__":
    settings = AtaraxAISettings()
    directories = AppDirectories.create_default(settings)

    logger = AtaraxAILogger().get_logger()
    model_manager = ModelsManager(directories, logger)

    models = model_manager.search_models("llama", limit=100)
    print(f"Found {len(models)} models matching 'llama'.")

    if models:
        model_to_download = random.choice(models)
        print(f"Selected model to download: {model_to_download}")

        # files = model_manager.list_available_files(model_to_download["id"])
        # print(f"Available files: {files}")

        # if model_to_download:
        #     filename = model_to_download.filename
        #     print(f"Downloading {filename} from {model_to_download.repo_id}...")
        #     task_id = model_manager.start_download_task(
        #         model_to_download.repo_id, filename, model_info=model_to_download
        #     )
        #     print(f"Download started with task ID: {task_id}")

        #     while True:
        #         status = model_manager.get_download_status(task_id)
        #         if status:
        #             # print(
        #             #     f"Status: {status['status']}, Progress: {status['progress']:.2%}"
        #             # )
        #             if status["status"] in [
        #                 ModelDownloadStatus.COMPLETED,
        #                 ModelDownloadStatus.FAILED,
        #             ]:
        #                 break
        #         time.sleep(2)

        #     downloaded = model_manager.list_downloaded_models()
        #     print(f"Downloaded models: {len(downloaded)}")
    else:
        print("No models found.")
