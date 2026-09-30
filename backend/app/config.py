 
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    generator_base_url: str = "http://localhost:11434/v1"
    generator_api_key: str = "ollama"
    generator_model: str = "llama3:latest"

    judge_base_url: str = "http://localhost:11434/v1"
    judge_api_key: str = "ollama"
    judge_model: str = "llama3:latest"


    # Embeddings (local, free)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    database_url: str = "sqlite:///./experiments.db"


settings = Settings()