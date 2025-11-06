from langchain_core.messages import (
    BaseMessage as LangChainBaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage as LangChainToolMessage,
)
from ag_ui.core import Message


def convert_agui_message_to_langchain(message: Message) -> LangChainBaseMessage:
    """
    Convert an AG-UI Message to a LangChain BaseMessage.

    Args:
        message: AG-UI Message (DeveloperMessage, SystemMessage, AssistantMessage, UserMessage, or ToolMessage)

    Returns:
        LangChain BaseMessage (SystemMessage, HumanMessage, AIMessage, or ToolMessage)
    """
    if message.role == "system" or message.role == "developer":
        # Both system and developer messages map to SystemMessage in LangChain
        return SystemMessage(
            content=message.content or "",
            id=message.id,
            name=message.name,
        )
    elif message.role == "user":
        return HumanMessage(
            content=message.content or "",
            id=message.id,
            name=message.name,
        )
    elif message.role == "assistant":
        # Convert tool_calls if present
        additional_kwargs = {}
        if hasattr(message, "tool_calls") and message.tool_calls:
            additional_kwargs["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return AIMessage(
            content=message.content or "",
            id=message.id,
            name=message.name,
            additional_kwargs=additional_kwargs,
        )
    elif message.role == "tool":
        return LangChainToolMessage(
            content=message.content,
            tool_call_id=message.tool_call_id,
            id=message.id,
        )
    else:
        raise ValueError(f"Unknown message role: {message.role}")


def convert_agui_messages_to_langchain(
    messages: list[Message],
) -> list[LangChainBaseMessage]:
    """
    Convert a list of AG-UI Messages to LangChain BaseMessages.

    Args:
        messages: List of AG-UI Messages

    Returns:
        List of LangChain BaseMessages
    """
    return [convert_agui_message_to_langchain(msg) for msg in messages]
