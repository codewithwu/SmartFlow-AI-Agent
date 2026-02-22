from app.agent.tools.calculator import calculator
from app.agent.tools.web_search import web_search
from app.agent.tools.weather import weather_query
from app.agent.tools.database import database_query


def get_all_tools():
    """Return all available tools for the agent."""
    return [calculator, web_search, weather_query, database_query]
