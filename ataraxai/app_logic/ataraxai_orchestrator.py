import uuid
from pathlib import Path
from typing import Any, List, Dict

from ataraxai import __version__, core_ai_py  # type: ignore
from ataraxai.app_logic.preferences_manager import PreferencesManager
from ataraxai.app_logic.utils.ataraxai_logger import ArataxAILogger
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import LlamaModelParams
from ataraxai.app_logic.utils.llama_config_manager import LlamaConfigManager
from ataraxai.app_logic.utils.rag_config_manager import RAGConfigManager
from ataraxai.app_logic.utils.whisper_config_manager import WhisperConfigManager
from platformdirs import user_config_dir, user_data_dir, user_cache_dir, user_log_dir

from ataraxai.app_logic.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager

from ataraxai.app_logic.modules.prompt_engine.context_manager import (
    ContextManager,
    TaskContext,
)

from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.app_logic.modules.prompt_engine.task_manager import TaskManager
from ataraxai.app_logic.modules.prompt_engine.chain_runner import ChainRunner


from ataraxai.app_logic.utils.config_schemas.llama_config_schema import (
    GenerationParams,
    LlamaModelParams,
)
from ataraxai.app_logic.utils.config_schemas.whisper_config_schema import (
    WhisperModelParams,
    WhisperTranscriptionParams,
)

from typing import Tuple

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"


def init_params(
    llama_model_params: LlamaModelParams,
    llama_generation_params: GenerationParams,
    whisper_model_params: WhisperModelParams,
    whisper_transcription_params: WhisperTranscriptionParams,
) -> Tuple[Any, Any, Any, Any]:
    project_dir = Path(__file__).resolve().parent.parent.parent

    llama_model_params_cc = core_ai_py.LlamaModelParams.from_dict(  # type: ignore
        llama_model_params.model_dump()
    )

    llama_generation_params_cc = core_ai_py.GenerationParams.from_dict(  # type: ignore
        llama_generation_params.model_dump()
    )

    whisper_model_params_cc = core_ai_py.WhisperModelParams.from_dict(  # type: ignore
        whisper_model_params.model_dump()
    )

    whisper_transcription_params_cc = core_ai_py.WhisperTranscriptionParams.from_dict(  # type: ignore
        whisper_transcription_params.model_dump()
    )

    return (
        llama_model_params_cc,
        llama_generation_params_cc,
        whisper_model_params_cc,
        whisper_transcription_params_cc,
    )  # type: ignore


def initialize_core_ai_service(llama_params, whisper_params):  # type: ignore
    core_ai_service = core_ai_py.CoreAIService()  # type: ignore
    core_ai_service.initialize_llama_model(llama_params)  # type: ignore
    core_ai_service.initialize_whisper_model(whisper_params)  # type: ignore
    return core_ai_service  # type: ignore


class AtaraxAIOrchestrator:

    def __init__(self):
        self.logger = ArataxAILogger()
        self._init_user_dirs()
        self.ataraxai_version = __version__
        self.app_setup_marker_file: Path = (
            self.app_config_dir
            / f".ataraxai_app_{self.ataraxai_version}_setup_complete"
        )
        is_app_first_launch = not self.app_setup_marker_file.exists()

        self._init_configs()

        print("Initializing Core Services...")
        (
            llama_model_params_cc,
            llama_generation_params_cc,
            whisper_model_params_cc,
            whisper_transcription_params_cc,
        ) = init_params(
            llama_model_params=self.llama_config_manager.get_llm_params(),
            llama_generation_params=self.llama_config_manager.get_generation_params(),
            whisper_model_params=self.whisper_config_manager.get_whisper_params(),
            whisper_transcription_params=self.whisper_config_manager.get_transcription_params(),
        )
        self.cpp_service = initialize_core_ai_service(  # type: ignore
            llama_params=llama_model_params_cc,
            whisper_params=whisper_model_params_cc
        )  # type: ignore

        db_path = self.app_data_dir / "chat_history.sqlite"
        self.db_manager = ChatDatabaseManager(db_path=db_path)

        self.rag_manager = AtaraxAIRAGManager(
            preferences_manager_instance=self.prefs_manager,
            app_data_root_path=self.app_data_dir,
        )
        print("Core Services Initialized.")

        print("Initializing Prompt Engine...")
        prompts_dir = Path("./prompts")
        prompts_dir.mkdir(exist_ok=True)

        self.prompt_manager = PromptManager(prompts_directory=prompts_dir)
        self.context_manager = ContextManager(
            config=self.rag_config_manager.get_config().model_dump(),
            rag_manager=self.rag_manager,
        )
        self.task_manager = TaskManager()

        self.chain_runner = ChainRunner(
            task_manager=self.task_manager,
            context_manager=self.context_manager,
            prompt_manager=self.prompt_manager,
            core_ai_service=self.cpp_service,  # type: ignore
            chat_db_manager=self.db_manager,
            rag_manager=self.rag_manager,
        )
        print("Prompt Engine Initialized.")

        if is_app_first_launch:
            self._perform_first_launch_setup()

        self._start_rag_monitoring()

    def run_task_chain(self, chain_definition: list, initial_user_query: str) -> Any:
        print(f"Orchestrator executing chain for query: '{initial_user_query}'")
        return self.chain_runner.run_chain(
            chain_definition=chain_definition, initial_user_query=initial_user_query
        )

    def _init_user_dirs(self):
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

    def _init_configs(self):
        self.prefs_manager = PreferencesManager(config_path=self.app_config_dir)
        self.llama_config_manager = LlamaConfigManager(config_path=self.app_config_dir)
        self.whisper_config_manager = WhisperConfigManager(
            config_path=self.app_config_dir
        )
        self.rag_config_manager = RAGConfigManager(config_path=self.app_config_dir)

    def _perform_first_launch_setup(self):
        print("Performing application-wide first-launch tasks...")
        self.app_setup_marker_file.touch(exist_ok=False)

    def _start_rag_monitoring(self):
        watched_dirs: List[str] = self.prefs_manager.get("rag_watched_directories", [])  # type: ignore
        self.rag_manager.start_file_monitoring(watched_dirs)

    def shutdown(self):
        print("Shutting down...")
        self.rag_manager.stop_file_monitoring()
        self.db_manager.close()
        if hasattr(self, 'cpp_service') and self.cpp_service: # type: ignore
            self.cpp_service.shutdown()  # type: ignore
