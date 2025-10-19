from typing import Literal, cast
from app.agent.react_agent.states import AgentState
from langgraph.runtime import Runtime
from app.agent.react_agent.context import AppContext
from app.agent.react_agent.tools import TOOLS
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)


def agent_node(state: AgentState, runtime: Runtime[AppContext]) -> AgentState:
    print("\n\nThinking...\n\n")
    messages = state.messages
    if not messages:
        raise Exception("The messages cannot be empty")
    llm_with_tools = runtime.context.llm.bind_tools(
        tools=TOOLS, parallel_tool_calls=False
    )
    results = llm_with_tools.invoke(messages)
    results = cast(AIMessage, results)
    return AgentState(messages=[results])


def router_function(
    state: AgentState, runtime: Runtime[AppContext]
) -> Literal["agent_node", "tool_node", "__end__"]:
    latest_msg = state.messages[-1]
    if not latest_msg:
        raise Exception("The messages cannot be empty")
    if isinstance(latest_msg, HumanMessage):
        return "agent_node"
    if isinstance(latest_msg, AIMessage):
        if latest_msg.tool_calls:
            return "tool_node"
        if "FURTHER THINKING" in latest_msg.content:
            return "agent_node"
        else:
            return "__end__"
