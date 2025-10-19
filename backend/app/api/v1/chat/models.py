from pydantic import BaseModel


class SendMessagePost(BaseModel):
    assistant_id: int
    message: str
