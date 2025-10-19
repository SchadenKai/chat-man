from datetime import datetime
import random
from typing import Annotated, Literal, Sequence, cast
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from backend.app.config import OPENAI_API_KEY


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
)


# Tools available


@tool
def get_weather_update(city: str, date: datetime) -> str:
    """
    Used to get the weather based on the city and the date
    Args:
        city: Name of the city and the coutry. ex. San Luis, Batangas, Philippines
        date: datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
    """
    temp = random.randint(-15, 40)
    weather: Literal["snowing", "raining", "sunny"] = "sunny"
    if temp >= -15 and temp < 0:
        weather = "snowing"
    elif temp >= 0 and temp < 25:
        weather = "raining"
    return f"It is {weather}: {temp} degrees celcius in {city} at {date}"


@tool
def get_name_of_user() -> str:
    """
    Get the name of the current user through accessing the database and session id
    """
    return "Name: Kairus Noah E. Tecson, Occupation: Trashtalker"


tools = [get_weather_update, get_name_of_user]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


def agent_node(state: AgentState) -> AgentState:
    print("\n\nThinking...\n\n")
    messages = state.messages
    if not messages:
        raise Exception("The messages cannot be empty")
    results = llm_with_tools.invoke(messages)
    results = cast(AIMessage, results)
    return AgentState(messages=[results])


# def observation_node(state: AgentState) -> AgentState:
#     """
#     Used to synthesize and understand the results of the tool call properly before thinking further
#     """
#     print("Observing")
#     messages = state.messages
#     if not messages:
#         raise Exception("The messages cannot be empty")
#     messages.append(
#         SystemMessage(
#             content="Based on the recent tool call, creat a proper observation on the results by sythensizing and properly understanding it."
#         )
#     )

#     results = llm_with_tools.invoke(messages)
#     results = cast(AIMessage, results)
#     return AgentState(messages=[results])


def router_function(state: AgentState) -> Literal["agent_node", "tool_node", "__end__"]:
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


graph = StateGraph(state_schema=AgentState)

few_shot_examples_condensed = [
    HumanMessage(
        content="Get weather for Manila, Philippines and Seoul, South Korea for October 18, 2025 at 2:00 PM. Include their populations."
    ),
    AIMessage(
        content="I'll get the weather for both cities, then add population data from my knowledge.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {"city": "Manila, Philippines", "date": "2025-10-18T14:00:00"},
                "id": "call_1",
            }
        ],
    ),
    ToolMessage(
        content="It is sunny: 32 degrees celsius in Manila, Philippines at 2025-10-18 14:00:00",
        tool_call_id="call_1",
    ),
    AIMessage(content="FURTHER THINKING - Manila done. Getting Seoul weather next."),
    AIMessage(
        content="Retrieving Seoul weather.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {"city": "Seoul, South Korea", "date": "2025-10-18T14:00:00"},
                "id": "call_2",
            }
        ],
    ),
    ToolMessage(
        content="It is raining: 15 degrees celsius in Seoul, South Korea at 2025-10-18 14:00:00",
        tool_call_id="call_2",
    ),
    AIMessage(
        content="""All weather data collected. Here's the complete information:

**Manila, Philippines:**
- Weather: Sunny, 32°C
- Population: ~1.8 million (city), ~14 million (metro area)

**Seoul, South Korea:**
- Weather: Raining, 15°C
- Population: ~9.7 million (city), ~25.6 million (metro area)"""
    ),
]

few_shot_examples = [
    # Example 1: Simple multi-city weather query
    HumanMessage(
        content="Get the weather for Tokyo, Japan and London, UK for today at 10:00 AM. Also tell me their populations."
    ),
    AIMessage(
        content="I need to retrieve weather information for two cities and provide population data. Let me start by getting the weather for Tokyo.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {"city": "Tokyo, Japan", "date": "2025-10-18T10:00:00"},
                "id": "call_1",
            }
        ],
    ),
    ToolMessage(
        content="It is sunny: 22 degrees celsius in Tokyo, Japan at 2025-10-18 10:00:00",
        tool_call_id="call_1",
    ),
    AIMessage(
        content="FURTHER THINKING - I have Tokyo's weather. Now I need to get London's weather before providing the complete answer with population data."
    ),
    AIMessage(
        content="Now retrieving weather for London.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {"city": "London, UK", "date": "2025-10-18T10:00:00"},
                "id": "call_2",
            }
        ],
    ),
    ToolMessage(
        content="It is raining: 12 degrees celsius in London, UK at 2025-10-18 10:00:00",
        tool_call_id="call_2",
    ),
    AIMessage(
        content="""I now have all the weather information. Let me provide the complete answer with population data from my knowledge:

**Tokyo, Japan:**
- Weather (Oct 18, 2025, 10:00 AM): Sunny, 22°C
- Population: ~14 million (city proper), ~37 million (metro area)

**London, UK:**
- Weather (Oct 18, 2025, 10:00 AM): Raining, 12°C
- Population: ~9 million (city proper), ~14 million (metro area)"""
    ),
    # Example 2: The actual target query - 4 cities with detailed info
    HumanMessage(
        content="Can you retrieve the following information: the weather information of each of these cities for today October 18, 2025 9:00am: San Luis, Batangas, Philippines; Batangas City, Philippines; Lemery, Batangas, Philippines; Vienna, Austria. I also want you to include the following information other than the weather for each of those cities: population, square meter size, estimated earnings, top market (You can get these information by just retrieving it from your pre-trained data. You will then need to perform a reasoning loop)"
    ),
    AIMessage(
        content="I understand the task. I need to: 1) Get weather data for 4 cities using the get_weather_update tool, and 2) Provide additional information (population, area, earnings, top market) from my knowledge base. I'll process each city systematically. Let me start with San Luis, Batangas.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {
                    "city": "San Luis, Batangas, Philippines",
                    "date": "2025-10-18T09:00:00",
                },
                "id": "call_1",
            }
        ],
    ),
    ToolMessage(
        content="It is sunny: 28 degrees celsius in San Luis, Batangas, Philippines at 2025-10-18 09:00:00",
        tool_call_id="call_1",
    ),
    AIMessage(
        content="FURTHER THINKING - San Luis weather retrieved: sunny, 28°C. I need to continue with the remaining 3 cities before compiling the full response with all the additional information."
    ),
    AIMessage(
        content="Retrieving weather for Batangas City next.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {
                    "city": "Batangas City, Philippines",
                    "date": "2025-10-18T09:00:00",
                },
                "id": "call_2",
            }
        ],
    ),
    ToolMessage(
        content="It is sunny: 29 degrees celsius in Batangas City, Philippines at 2025-10-18 09:00:00",
        tool_call_id="call_2",
    ),
    AIMessage(
        content="FURTHER THINKING - Batangas City weather: sunny, 29°C. Moving to Lemery next. 2 cities done, 2 remaining."
    ),
    AIMessage(
        content="Getting weather for Lemery, Batangas.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {
                    "city": "Lemery, Batangas, Philippines",
                    "date": "2025-10-18T09:00:00",
                },
                "id": "call_3",
            }
        ],
    ),
    ToolMessage(
        content="It is sunny: 27 degrees celsius in Lemery, Batangas, Philippines at 2025-10-18 09:00:00",
        tool_call_id="call_3",
    ),
    AIMessage(
        content="FURTHER THINKING - Lemery weather: sunny, 27°C. One more city remaining: Vienna, Austria. This will have cooler weather due to its location."
    ),
    AIMessage(
        content="Finally, retrieving weather for Vienna, Austria.",
        tool_calls=[
            {
                "name": "get_weather_update",
                "args": {"city": "Vienna, Austria", "date": "2025-10-18T09:00:00"},
                "id": "call_4",
            }
        ],
    ),
    ToolMessage(
        content="It is raining: 11 degrees celsius in Vienna, Austria at 2025-10-18 09:00:00",
        tool_call_id="call_4",
    ),
    AIMessage(
        content="""FURTHER THINKING - All weather data collected. Vienna: raining, 11°C. Now I have all 4 weather readings. Let me compile the complete response with all requested information including population, area in square meters, estimated earnings, and top market for each city using my pre-trained knowledge."""
    ),
    AIMessage(
        content="""Here's the complete information for all four cities as of October 18, 2025 at 9:00 AM:

**San Luis, Batangas, Philippines:**
- Weather: Sunny, 28°C
- Population: ~34,000
- Area: 38,400,000 m² (38.4 km²)
- Estimated Earnings: ₱150-200 million annually (municipal revenue)
- Top Market: Agriculture (sugarcane, rice, livestock, fishing)

**Batangas City, Philippines:**
- Weather: Sunny, 29°C
- Population: ~351,000
- Area: 282,480,000 m² (282.48 km²)
- Estimated Earnings: ₱2.5-3 billion annually (major industrial hub)
- Top Market: Manufacturing, port operations, oil refining, maritime trade

**Lemery, Batangas, Philippines:**
- Weather: Sunny, 27°C
- Population: ~109,000
- Area: 71,500,000 m² (71.5 km²)
- Estimated Earnings: ₱300-400 million annually (municipal revenue)
- Top Market: Agriculture (coffee, sugarcane, rice), tourism (Taal Lake proximity)

**Vienna, Austria:**
- Weather: Raining, 11°C
- Population: ~1.9 million (city), ~2.6 million (metro area)
- Area: 414,650,000 m² (414.65 km²)
- Estimated Earnings: ~€95 billion GDP annually (~$100 billion USD)
- Top Market: Professional services, international organizations, finance, technology, tourism"""
    ),
]

init_state = AgentState(
    messages=[
        SystemMessage(
            content="You are a helpful agent. You have the ability to do ReACT LLM flow where you can iterate between tool calls, reasoning to yourself, and exting with the final answer. If you need further thinking rather than just executing a tool, you can reply 'FURTHER THINKING' to trigger a reasoining loop"
        ),
        *few_shot_examples,
        AIMessage(
            content="Can you retrieve the following information the weather information of each of these city for today October 18, 2025 9:00am: San Luis, Batangas, Philippines; Batangas City, Philippines; Lemery, Batangas, Philippines; Vienna, Austria. I also want you to include the following information other than the weather for each of those cities: population, square meter size, estimated earnings, top market (You can get these information by just retrieving it from your pre-trained data. You will then need to perform a reasoinig loop)"
        ),
    ]
)

graph.add_node("agent_node", agent_node)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
# graph.add_node("observation_node", observation_node)
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", router_function)
# graph.add_edge("tool_node", "observation_node")
# graph.add_edge("observation_node", "agent_node")
graph.add_edge("tool_node", "agent_node")

agent = graph.compile()
for stream_mode, content in agent.stream(
    init_state, stream_mode=["updates", "messages"]
):
    # print("---- RAW RESULTS ----")
    if stream_mode == "updates":
        for node, state in content.items():
            print(node)
    if stream_mode == "messages":
        # print(content)
        msg_chunk, metadata = content
        if msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
    # for node, state in event.items():
    #     print(f"Node: {node}")
    #     print(f"Messages: {[msg.content for msg in state['messages']]}")
    #     print("---")

# for message_chunk, metadata in agent.stream(init_state, stream_mode=["messages"]):
#     if message_chunk.content:
#         print(message_chunk.content, end="", flush=True)
