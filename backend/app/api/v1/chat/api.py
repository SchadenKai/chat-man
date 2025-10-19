from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.v1.chat.models import SendMessagePost
from app.agent.react_agent.graph import call_agent

router = APIRouter(prefix="/chat")


@router.post("/send-message")
def send_message(req: SendMessagePost):
    return StreamingResponse(call_agent(req.message))
