 
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    generator_base_url: str = "http://localhost:11434/v1"
    generator_api_key: str = "ollama"
    generator_model: str = "llama3:latest"
    generator_min_interval_s: float = 0.0
    generator_max_tokens: int = 256

    judge_base_url: str = "http://localhost:11434/v1"
    judge_api_key: str = "ollama"
    judge_model: str = "llama3:latest"
    judge_min_interval_s: float = 0.0  # pause between judge calls (set > 0 for rate-limited free tiers)


    # Embeddings (local, free)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    database_url: str = "sqlite:///./experiments.db"
    cors_origins: str = "http://localhost:5173"
    ask_per_ip_per_hour: int = 5
    ask_daily_cap: int = 100

    admin_token: str = ""  # empty disables the optimizer job endpoints
    job_max_budget: int = 250
    job_max_experiments: int = 3
    job_approval_timeout_s: int = 900


settings = Settings()