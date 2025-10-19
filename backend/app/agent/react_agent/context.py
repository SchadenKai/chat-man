from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import OPENAI_API_KEY


class AppContext(BaseModel):
    system_prompt: str = Field(
        description="The system prompt to use for the agent's interactions."
    )
    max_iterations: int = Field(
        default=20, description="Maximum number of graph flow iteration"
    )
    llm: BaseChatModel = Field(
        default=None,
        description="LLM Chat Model class to call the LLM",
    )
