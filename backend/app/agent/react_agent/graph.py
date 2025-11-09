from typing import AsyncGenerator, cast
import uuid

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, BaseMessageChunk, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel

from app.agent.react_agent.tools import TOOLS
from app.agent.react_agent.states import AgentState
from app.agent.react_agent.nodes import agent_node, router_function
from app.agent.react_agent.prompts import REACT_AGENT_SYSTEM_PROMPT
from app.agent.react_agent.context import AppContext
from ag_ui.encoder import EventEncoder
from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    BaseEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallChunkEvent,
    ToolCallResultEvent
)

graph = StateGraph(state_schema=AgentState)

graph.add_node("agent_node", agent_node)
tool_node = ToolNode(tools=TOOLS)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", router_function)
graph.add_edge("tool_node", "agent_node")

checkpointer = InMemorySaver()

agent = graph.compile(checkpointer=checkpointer)


async def call_agent(
    human_message: str,
    llm: BaseChatModel,
    encoder: EventEncoder,
    thread_id: str = "5dc2ee5a-a752-45a4-9e65-650d06cb118a",
    run_id: str = "5dc2ee5a-a752-45a4-9e65-650d06cb118a",
) -> AsyncGenerator[BaseEvent, None]:
    # Use provided thread_id and run_id from AG-UI, or generate them
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    if run_id is None:
        run_id = str(uuid.uuid4())

    # for checkpoint
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    yield encoder.encode(
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            run_id=run_id,
            thread_id=thread_id,
        )
    )

    messages = []
    # Check if there is an existing agent state
    # If not, add the system prompt
    existing_agent_state = await agent.aget_state(config=config)
    if not existing_agent_state.values:
        messages.append(SystemMessage(content=REACT_AGENT_SYSTEM_PROMPT))
    messages.append(HumanMessage(content=human_message))

    init_state = AgentState(messages=messages)
    runtime_context_config = AppContext(
        llm=llm, system_prompt=REACT_AGENT_SYSTEM_PROMPT
    )

    # Generate a message ID for the assistant's response
    message_id = str(uuid.uuid4())
    message_started = False

    async for stream_mode, content in agent.astream(
        init_state,
        config=config,
        context=runtime_context_config,
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "updates":
          for node_name, state in content.items():
              if node_name == "tool_node":
                # Get the last message from the state which should be the tool result
                if state.get("messages"):
                    last_message = state["messages"][-1]
                    print("Tool call results: ", last_message)
                    yield encoder.encode(
                        ToolCallResultEvent(
                            type=EventType.TOOL_CALL_RESULT,
                            content=last_message.content,
                            tool_call_id=last_message.tool_call_id,
                            message_id=message_id
                        )
                    )
        elif stream_mode == "messages":
            msg_chunk, metadata = content
            node_name = metadata["langgraph_node"]
            print("EXECUTING NODE: ", node_name)
            msg_chunk = cast(BaseMessageChunk, msg_chunk)
            if node_name == "agent_node":
                msg_chunk = cast(AIMessageChunk, msg_chunk)
                if msg_chunk.tool_call_chunks:
                    for tool_call in msg_chunk.tool_calls:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args", {})
                        tool_id = tool_call.get("id", "")
                        if not tool_name or not tool_id:
                            continue
                        print("Tool name: ", tool_name)
                        print("Tool args: ", tool_args)
                        print("Tool id: ", tool_id)
                        yield encoder.encode(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_name=tool_name,
                                tool_call_id=tool_id
                            )
                        )
                        yield encoder.encode(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=tool_id,
                                delta=str(tool_args)
                            )
                        )
                        yield encoder.encode(
                            ToolCallEndEvent(
                                type=EventType.TOOL_CALL_END,
                                tool_call_id=tool_id
                            )
                        )
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