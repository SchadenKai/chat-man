from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from langchain_core.language_models import BaseChatModel
from app.agent.react_agent.graph import call_agent
from ag_ui.encoder import EventEncoder
from ag_ui.core import RunAgentInput

from app.agent.factory import get_default_llm

router = APIRouter(prefix="/chat")


@router.post("/send-message")
async def send_message(
    input_data: RunAgentInput, llm: BaseChatModel = Depends(get_default_llm)
):
    """
    AG-UI compatible endpoint that accepts RunAgentInput and streams back AG-UI events.

    Args:
        input_data (RunAgentInput): AG-UI standard input containing thread_id, run_id, messages, etc.

    Returns:
        StreamingResponse: Server-sent events stream with agent execution updates
    """
    encoder = EventEncoder()

    # Extract the user message from the messages array
    user_message = ""
    if input_data.messages and len(input_data.messages) > 0:
        # Get the last message (most recent user message)
        last_message = input_data.messages[-1]
        user_message = last_message.content if hasattr(last_message, "content") else ""

    return StreamingResponse(
        call_agent(
            human_message=user_message,
            llm=llm,
            encoder=encoder,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
        ),
        media_type="text/event-stream",
    )
