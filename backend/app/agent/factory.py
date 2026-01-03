from langchain_core.language_models import BaseChatModel

from app.agent.llm import LLMFactory
from app.core.config import settings


def get_default_llm() -> BaseChatModel:
    llm = LLMFactory(
        model_name="gpt-4o-mini",
        model_provider="openai",
        temperature=0.3,
        max_retries=1,
        api_key=settings.openai_api_key,
    )
    return llm.create_chat_model()
