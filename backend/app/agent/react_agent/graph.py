from typing import cast
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


async def call_agent(human_message: str):
    SYSTEM_PROMPT = "You are a helpful agent. You have the ability to do ReACT LLM flow where you can iterate between tool calls, reasoning to yourself, and exting with the final answer. If you need further thinking rather than just executing a tool, you can reply 'FURTHER THINKING' to trigger a reasoining loop"
    init_state = AgentState(
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            *few_shot_examples,
            AIMessage(content=human_message),
        ]
    )
    runtime_context_config = AppContext(llm=llm, system_prompt=SYSTEM_PROMPT)

    for stream_mode, content in agent.stream(
        init_state, context=runtime_context_config, stream_mode=["updates", "messages"]
    ):
        if stream_mode == "messages":
            msg_chunk, metadata = content
            msg_chunk = cast(BaseMessage, msg_chunk)
            if msg_chunk.content:
                yield msg_chunk.content


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
