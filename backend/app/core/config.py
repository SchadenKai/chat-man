from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: Optional[str]
    nebius_api_key: Optional[str]
    brave_api_key: Optional[str]
    gemini_api_key: Optional[str]


settings = Settings()
