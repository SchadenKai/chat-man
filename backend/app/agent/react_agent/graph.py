from typing import cast
import uuid
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode


from app.agent.react_agent.tools import TOOLS
from app.agent.react_agent.states import AgentState
from app.agent.react_agent.nodes import agent_node, router_function
from app.agent.react_agent.prompts import few_shot_examples
from app.agent.react_agent.context import AppContext
from app.config import OPENAI_API_KEY
from ag_ui.encoder import EventEncoder
from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
)

graph = StateGraph(state_schema=AgentState)

graph.add_node("agent_node", agent_node)
tool_node = ToolNode(tools=TOOLS)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", router_function)
graph.add_edge("tool_node", "agent_node")

agent = graph.compile()


async def call_agent(human_message: str, encoder: EventEncoder, thread_id: str = None, run_id: str = None):
    # Use provided thread_id and run_id from AG-UI, or generate them
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    if run_id is None:
        run_id = str(uuid.uuid4())

    yield encoder.encode(
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            run_id=run_id,
            thread_id=thread_id,
        )
    )

    SYSTEM_PROMPT = "You are a helpful agent. You have the ability to do ReACT LLM flow where you can iterate between tool calls, reasoning to yourself, and exting with the final answer. If you need further thinking rather than just executing a tool, you can reply 'FURTHER THINKING' to trigger a reasoining loop"
    init_state = AgentState(
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            *few_shot_examples,
            AIMessage(content=human_message),
        ]
    )
    runtime_context_config = AppContext(llm=llm, system_prompt=SYSTEM_PROMPT)

    # Generate a message ID for the assistant's response
    message_id = str(uuid.uuid4())
    message_started = False

    for stream_mode, content in agent.stream(
        init_state, context=runtime_context_config, stream_mode=["updates", "messages"]
    ):
        if stream_mode == "messages":
            msg_chunk, metadata = content
            msg_chunk = cast(BaseMessage, msg_chunk)
            if msg_chunk.content:
                # Send TEXT_MESSAGE_START event on first chunk
                if not message_started:
                    yield encoder.encode(
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START,
                            message_id=message_id,
                            role="assistant",
                        )
                    )
                    message_started = True

                # Send TEXT_MESSAGE_CONTENT event for each chunk
                yield encoder.encode(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=str(msg_chunk.content),
                    )
                )

    # Send TEXT_MESSAGE_END event after streaming completes
    if message_started:
        yield encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id,
            )
        )

    yield encoder.encode(
        RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            run_id=run_id,
            thread_id=thread_id,
        )
    )


# for stream_mode, content in agent.stream(
#     init_state, stream_mode=["updates", "messages"]
# ):
#     # print("---- RAW RESULTS ----")
#     if stream_mode == "updates":
#         for node, state in content.items():
#             print(node)
#     if stream_mode == "messages":
#         # print(content)
#         msg_chunk, metadata = content
#         if msg_chunk.content:
#             print(msg_chunk.content, end="", flush=True)
# for node, state in event.items():
#     print(f"Node: {node}")
#     print(f"Messages: {[msg.content for msg in state['messages']]}")
#     print("---")

# for message_chunk, metadata in agent.stream(init_state, stream_mode=["messages"]):
#     if message_chunk.content:
#         print(message_chunk.content, end="", flush=True)
