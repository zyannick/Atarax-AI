from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from platformdirs import user_config_dir, user_data_dir, user_cache_dir, user_log_dir
from ataraxai.praxis.utils.ataraxai_settings import AtaraxAISettings
from ataraxai import __version__


@dataclass
class AppDirectories:
    config: Path
    data: Path
    cache: Path
    logs: Path

    @classmethod
    def create_default(cls, settings: AtaraxAISettings) -> "AppDirectories":
        """
        Creates an instance of AppDirectories with default paths for config, data, cache, and logs
        using the user's operating system conventions. The directories are created if they do not exist.

        Returns:
            AppDirectories: An instance with initialized and created directory paths.
        """
        dirs = cls(
            config=Path(
                user_config_dir(
                    appname=settings.app_name,
                    appauthor=settings.app_author,
                    version=__version__,
                )
            ),
            data=Path(
                user_data_dir(
                    appname=settings.app_name,
                    appauthor=settings.app_author,
                    version=__version__,
                )
            ),
            cache=Path(
                user_cache_dir(
                    appname=settings.app_name,
                    appauthor=settings.app_author,
                    version=__version__,
                )
            ),
            logs=Path(
                user_log_dir(
                    appname=settings.app_name,
                    appauthor=settings.app_author,
                    version=__version__,
                )
            ),
        )
        dirs.create_directories()
        return dirs

    def create_directories(self) -> None:
        """
        Creates the necessary directories for configuration, data, cache, and logs.

        This method iterates over the predefined directory paths (self.config, self.data, self.cache, self.logs)
        and ensures that each directory exists by creating it if it does not already exist. Parent directories
        are also created as needed.

        Returns:
            None
        """
        for directory in [self.config, self.data, self.cache, self.logs]:
            directory.mkdir(parents=True, exist_ok=True)
