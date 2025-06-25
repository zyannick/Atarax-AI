from pathlib import Path
from ataraxai.app_logic.preferences_manager import PreferencesManager
from ataraxai.app_logic.utils.llama_config_manager import LlamaConfigManager
from ataraxai.app_logic.utils.whisper_config_manager import WhisperConfigManager
from ataraxai.app_logic.utils.rag_config_manager import RAGConfigManager
from ataraxai import core_ai_py # type: ignore [attr-defined]
from platformdirs import user_data_dir, user_config_dir, user_cache_dir, user_log_dir
from ataraxai.app_logic.utils.ataraxai_logger import ArataxAILogger


APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"


class AtaraxAIOrchestrator:
    """
    The AtaraxAIOrchestrator class is responsible for managing the orchestration of the AtaraxAI application.
    It handles the initialization and coordination of various components within the application.
    """

    def __init__(self):
        """
        Initializes the AtaraxAIOrchestrator instance.
        """
        self.logger = ArataxAILogger()

        # self._init_user_dirs()

        # self.ataraxai_version = __version__
        # self.app_setup_marker_file = (
        #     self.app_config_dir / ".ataraxai_app_"
        #     + self.ataraxai_version
        #     + "_setup_complete"
        # )
        # is_app_first_launch = not self.app_setup_marker_file.exists()

        # self._init_configs()

        
        

        # if is_app_first_launch:
        #     print("MainApplication: First application launch detected.")
        #     self._perform_application_first_launch_setup()
        # else:
        #     print("MainApplication: Subsequent application launch.")

        # self.rag_manager = AtaraxAIRAGManager(
        #     core_ai_service_instance=self.cpp_service,
        #     preferences_manager_instance=self.prefs_manager,
        #     app_data_root_path=self.app_data_dir,
        # )

        # self._load_models()
        # self._start_rag_monitoring()

        print(f"{APP_NAME} initialized.")

    def _init_user_dirs(self):
        """
        Initializes user-specific directories for configuration, data, cache, and logs.
        """
        self.app_config_dir = Path(
            user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
        )
        self.app_data_dir = Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
        self.app_cache_dir = Path(
            user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
        )
        self.app_log_dir = Path(user_log_dir(appname=APP_NAME, appauthor=APP_AUTHOR))

        self.app_config_dir.mkdir(parents=True, exist_ok=True)
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        self.app_cache_dir.mkdir(parents=True, exist_ok=True)
        self.app_log_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"User directories initialized at: {self.app_config_dir}, {self.app_data_dir}, {self.app_cache_dir}, {self.app_log_dir}"
        )

    def _init_configs(self):
        self.prefs_manager = PreferencesManager(config_path=self.app_config_dir)
        self.llama_config_manager = LlamaConfigManager(config_path=self.app_config_dir)
        self.whisper_config_manager = WhisperConfigManager(
            config_path=self.app_config_dir
        )
        self.rag_config_manager = RAGConfigManager(config_path=self.app_config_dir)

    def _perform_first_launch_setup(self):
        print("MainApplication: Performing application-wide first-launch tasks...")

        print(
            "Welcome to AtaraxAI! Please complete the initial setup in the application settings to select models and data sources."
        )

        try:
            self.app_setup_marker_file.touch(exist_ok=False)
            print(
                f"Application first-launch setup complete. Marker created: {self.app_setup_marker_file}"
            )
        except Exception as e:
            print(f"ERROR: Could not create application first-launch marker: {e}")

    def _load_models(self):

        llama_path = self.prefs_manager.get("llama_model_path")
        if llama_path:
            print(f"Attempting to load LLaMA from: {llama_path}")

        pass

    def _start_rag_monitoring(self):
        watched_dirs = self.prefs_manager.get("rag_watched_directories", [])
        self.rag_manager.start_file_monitoring(watched_dirs)

    def shutdown(self):
        print("MainApplication: Shutting down...")
        if self.rag_manager:
            self.rag_manager.stop_file_monitoring()
        core_ai_py.CoreAIService.free_global_backends()
        print("MainApplication: Shutdown complete.")


if __name__ == "__main__":
    orchestrator = AtaraxAIOrchestrator()
    try:
        print("AtaraxAI is running. Press Ctrl+C to exit.")
        while True:
            pass  # Keep the application running
    except KeyboardInterrupt:
        print("Exiting AtaraxAI...")
    finally:
        orchestrator.shutdown()
