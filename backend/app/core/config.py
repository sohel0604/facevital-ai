"""
FaceVital AI — Application Configuration
=========================================
Central configuration using Pydantic Settings.
All secrets and tunables come from environment variables.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Application ---
    app_name: str = "FaceVital AI"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # --- Backend ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_workers: int = 1

    # --- Database ---
    # Default to zero-config local SQLite. Set to postgresql+asyncpg:// for production/Docker.
    database_url: str = "sqlite+aiosqlite:///facevital.db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "facevital"
    postgres_user: str = "facevital_user"
    postgres_password: str = "change_me_in_production"

    # --- Security ---
    secret_key: str = "change-this-to-a-random-secret-key-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- Model ---
    model_weights_dir: str = "./models/weights"
    model_version: str = "v1.0.0-placeholder"

    # --- rPPG ---
    rppg_window_seconds: float = 10.0
    rppg_overlap: float = 0.5
    rppg_min_hr: float = 40.0
    rppg_max_hr: float = 180.0
    signal_quality_threshold: float = 0.4

    # --- Rate Limiting ---
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # --- Observability ---
    enable_structured_logging: bool = True

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def db_url_sync(self) -> str:
        """Synchronous database URL for migrations / utilities."""
        url = self.db_url
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "")
        if "+aiosqlite" in url:
            return url.replace("+aiosqlite", "")
        return url

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
