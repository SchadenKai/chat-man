import random
from typing import Callable, List, Literal
from datetime import datetime


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


TOOLS: List[Callable[..., any]] = [get_weather_update, get_name_of_user]
