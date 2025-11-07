import random
from typing import Callable, List, Literal
from datetime import datetime
from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel


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


def get_name_of_user() -> str:
    """
    Get the name of the current user through accessing the database and session id
    """
    return "Name: Kairus Noah E. Tecson, Occupation: Trashtalker"


def do_reasoning(runtime: ToolRuntime) -> str:
    """
    Perform intermediate reasoning step which will call an LLM that has access to the whole conversation history.
    """
    print("Performing reasoning step...")
    llm = runtime.context.llm
    messages = runtime.state.messages

    if not messages:
        raise Exception("The messages cannot be empty")
    if not llm:
        raise Exception("LLM is not found in the context")
    if not isinstance(messages, list):
        raise Exception("Messages should be a list of BaseMessage")
    if not isinstance(llm, BaseChatModel):
        raise Exception("LLM should be an instance of BaseChatModel")

    messages = [
        HumanMessage(
            content=f"Based on the previous conversation, please reason and provide your next response. ## Chat Conversation: {str(messages)}"
        )
    ]
    response: AIMessage = llm.invoke(messages)
    response = response.content
    return response


TOOLS: List[Callable[..., any]] = [get_weather_update, get_name_of_user, do_reasoning]
