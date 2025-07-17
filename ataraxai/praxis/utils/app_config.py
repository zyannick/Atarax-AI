from __future__ import annotations
from dataclasses import dataclass



@dataclass
class AppConfig:
    database_filename: str = "chat_history.sqlite"
    prompts_directory: str = "./prompts"
    setup_marker_filename: str = ".ataraxai_app_{version}_setup_complete"

    def get_setup_marker_filename(self, version: str) -> str:
        """
        Generate the setup marker filename for a specific version.

        Args:
            version (str): The version string to include in the filename.

        Returns:
            str: The formatted setup marker filename for the given version.
        """
        return self.setup_marker_filename.format(version=version)
