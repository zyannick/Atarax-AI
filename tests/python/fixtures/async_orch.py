import asyncio
from pathlib import Path

from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
)
from ataraxai.praxis.modules.benchmarker.benchmarker import BenchmarkQueueManager
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.models_manager.models_manager import (
    ModelsManager,
)
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.ataraxai_settings import AtaraxAISettings
from ataraxai.praxis.utils.background_task_manager import BackgroundTaskManager
from ataraxai.praxis.utils.chat_manager import ChatManager
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager
from ataraxai.praxis.utils.services import Services
from ataraxai.praxis.utils.setup_manager import SetupManager
from ataraxai.praxis.utils.vault_manager import VaultManager


async def setup_async_orchestrator(temp_dir_path: Path) -> AtaraxAIOrchestrator:
    app_config = AppConfig()
    settings = AtaraxAISettings()

    base_path = temp_dir_path
    directories = AppDirectories(
        config=base_path / "config",
        data=base_path / "data",
        cache=base_path / "cache",
        logs=base_path / "logs",
    )
    await asyncio.to_thread(directories.create_directories)

    logger = AtaraxAILogger(log_dir=directories.logs).get_logger()
    vault_manager = VaultManager(
        salt_path=str(directories.data / "vault.salt"),
        check_path=str(directories.data / "vault.check"),
    )
    setup_manager = SetupManager(directories, app_config, logger)
    config_manager = ConfigurationManager(directories.config, logger)
    core_ai_manager = CoreAIServiceManager(config_manager, logger)
    db_manager = ChatDatabaseManager(
        db_path=directories.data / app_config.database_filename
    )
    chat_context = ChatContextManager(
        db_manager=db_manager, vault_manager=vault_manager
    )
    chat_manager = ChatManager(
        db_manager=db_manager, logger=logger, vault_manager=vault_manager
    )
    background_task_manager = BackgroundTaskManager()
    models_manager = ModelsManager(
        directories=directories,
        logger=logger,
        background_task_manager=background_task_manager,
    )

    benchmark_queue_manager = BenchmarkQueueManager(
        logger=logger,
        max_concurrent=1,
        persistence_file=directories.data / "benchmark_jobs.json"
    )

    services = Services(
        directories=directories,
        logger=logger,
        db_manager=db_manager,
        chat_context=chat_context,
        chat_manager=chat_manager,
        config_manager=config_manager,
        app_config=app_config,
        vault_manager=vault_manager,
        models_manager=models_manager,
        core_ai_service_manager=core_ai_manager,
        background_task_manager=background_task_manager,
        benchmark_queue_manager=benchmark_queue_manager,
    )

    orchestrator = AtaraxAIOrchestrator(
        settings=settings, setup_manager=setup_manager, services=services, logger=logger
    )

    await orchestrator.initialize()
    return orchestrator
