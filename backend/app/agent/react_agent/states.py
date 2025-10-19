from typing import Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )
