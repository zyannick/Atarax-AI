from pydantic_settings import BaseSettings


class AtaraxAISettings(BaseSettings):
    app_name: str = "AtaraxAI"
    app_author: str = "AtaraxAI"
    # database_filename: str = "chat_history.sqlite"
    log_level: str = "INFO"
    max_retries: int = 3

    class Config:
        env_prefix = "ATARAXAI_"
        case_sensitive = False
