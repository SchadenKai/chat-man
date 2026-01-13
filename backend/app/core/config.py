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
    vector_dimension: Optional[int] = 1536

    # Change default value as needed through .env file
    milvus_uri: Optional[str] = "http://windows-server:19530"
    milvus_db_name: Optional[str] = "chat_man_db"
    collection_name: Optional[str] = "chat_rag"
    postgres_host: Optional[str] = "http://windows-server:5432"
    postgres_db_name: Optional[str] = "chat_man_db"
    postgres_user: Optional[str] = "root"
    postgres_password: Optional[str] = "password"


settings = Settings()
