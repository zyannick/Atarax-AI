from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.ataraxai_logger import ArataxAILogger
from ataraxai import __version__


class SetupManager:

    def __init__(
        self, directories: AppDirectories, config: AppConfig, logger: ArataxAILogger
    ):
        """
        Initializes the orchestrator with application directories, configuration, and logger.

        Args:
            directories (AppDirectories): Object containing paths to application directories.
            config (AppConfig): Application configuration object.
            logger (ArataxAILogger): Logger instance for logging application events.
        """
        self.directories = directories
        self.config = config
        self.logger = logger
        self.version = __version__
        self._marker_file = (
            self.directories.config
            / self.config.get_setup_marker_filename(self.version)
        )

    def is_first_launch(self) -> bool:
        """
        Checks if this is the first launch by verifying the existence of a marker file.

        Returns:
            bool: True if the marker file does not exist (indicating first launch), False otherwise.
        """
        return not self._marker_file.exists()

    def perform_first_launch_setup(self) -> None:
        """
        Performs the initial setup required during the application's first launch.

        This method checks if the application is being launched for the first time.
        If so, it executes the necessary setup steps and creates a marker to indicate
        that the setup has been completed. If the setup has already been performed,
        it skips the process. Logs the progress and any errors encountered.

        Raises:
            Exception: If any error occurs during the setup process.
        """
        if not self.is_first_launch():
            self.logger.info("Skipping first launch setup - already completed")
            return

        self.logger.info("Performing first launch setup...")
        try:
            self._create_setup_marker()
            self.logger.info("First launch setup completed successfully")
        except Exception as e:
            self.logger.error(f"First launch setup failed: {e}")
            raise

    def _create_setup_marker(self) -> None:
        """
        Creates a marker file to indicate that the setup process has been completed.

        This method attempts to create the marker file specified by `self._marker_file`.
        If the file already exists, a FileExistsError will be raised due to `exist_ok=False`.
        """
        self._marker_file.touch(exist_ok=False)
