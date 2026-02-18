from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Medgent Backend"
    app_env: str = "dev"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./medgent.db"
    api_key: str = "dev-local-key"
    artifact_dir: str = "./data/artifacts"
    inference_provider: Literal["mock", "medgemma"] = "mock"
    medgemma_base_url: str = "http://127.0.0.1:9000"
    medgemma_timeout_seconds: float = 30.0
    worker_poll_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
