from typing import Literal
from langchain_core.messages import BaseMessage, AIMessage
from langchain_nebius import ChatNebius
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

_supported_model_providers = Literal["openai", "nebius"]

class LLMFactory:
    def __init__(
        self,
        model_name: str,
        model_provider: _supported_model_providers,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        streaming: bool | None = None,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.temperature = temperature
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.streaming = streaming

    def prepare_base_kwargs(self) -> dict[str, any]:
        base_kwargs = {
            "model": self.model_name,
        }
        if self.max_tokens:
            base_kwargs["max_tokens"] = self.max_tokens
        if self.temperature:
            base_kwargs["temperature"] = self.temperature
        if self.timeout:
            base_kwargs["timeout"] = self.timeout
        if self.streaming:
            base_kwargs["streaming"] = self.streaming
        if self.max_retries:
            base_kwargs["max_retries"] = self.max_retries
        if self.reasoning_effort:
            base_kwargs["reasoning_effort"] = self.reasoning_effort
        return base_kwargs
    
    def create_chat_model(self) -> BaseChatModel:
        base_kwargs = self.prepare_base_kwargs()
        print("Base Kwargs: ", base_kwargs)
        if self.model_provider == "openai":
            return ChatOpenAI(**base_kwargs)
        elif self.model_provider == "nebius":
            return ChatNebius(**base_kwargs)