from pydantic import BaseModel
from ag_ui.core import RunAgentInput


class SendMessagePost(BaseModel):
    assistant_id: int
    message: str
