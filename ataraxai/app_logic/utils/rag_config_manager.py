import yaml
from pathlib import Path
from .config_schemas.rag_config_schema import (
    RAGConfig,
)
from typing_extensions import Optional


RAG_CONFIG_FILENAME = "rag_config.yaml"


class RAGConfigManager:
    """
    Manages the configuration for the RAG (Retrieval-Augmented Generation) system.
    Loads, saves, and updates the RAG configuration settings.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path / RAG_CONFIG_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config: RAGConfig = self._load_or_initialize()

    def _load_or_initialize(self) -> RAGConfig:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return RAGConfig(**yaml.safe_load(f))
            except Exception as e:
                print(f"[ERROR] Failed to load YAML config: {e}")
        print(f"[INFO] Creating default RAG config at: {self.config_path}")
        default = RAGConfig()
        self._save(default)
        return default

    def _save(self, config: Optional[RAGConfig] = None):
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config.model_dump() if config else self.config.model_dump(),
                f,
                default_flow_style=False,
            )

    def get_config(self) -> RAGConfig:
        return self.config

    def update_config(self, new_config: RAGConfig):
        self.config = new_config
        self._save(new_config)

    def reload(self):
        self.config = self._load_or_initialize()
