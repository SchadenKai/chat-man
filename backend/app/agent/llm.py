from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI


class LLM:
    def __init__(
        self,
        model_name: str,
        model_provider: str,
        temperature: int,
        max_token: int,
        timeout: int,
        response_format: any | None = None,
        base_url: str | None = None,
    ):
        self.model_name = (model_name,)
        self.model_provider = model_provider
        self.temperature = temperature
        self.base_url = base_url
        self.max_token = max_token
        self.timeout = timeout
        self.response_format = response_format

    def create_chat_model(self):
        if self.model_provider == "openai":
            return ChatOpenAI

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        llm = self.create_chat_model("openai")
        result = llm.invoke(messages)
        return result
