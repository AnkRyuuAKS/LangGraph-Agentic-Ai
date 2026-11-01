import os
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from e2b_code_interpreter import Sandbox
from langchain.agents import create_agent

load_dotenv()

# ---------------- LLM ----------------

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

# ---------------- SEARCH TOOL ----------------

search_tool = DuckDuckGoSearchRun()


# ---------------- WEATHER TOOL ----------------

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a given city.
    """

    try:
        # Step 1: Convert city name into coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=15
        )

        geo_response.raise_for_status()
        geo_data = geo_response.json()

        results = geo_data.get("results", [])

        if not results:
            return f"Location not found: {city}"

        location = results[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # Step 2: Fetch current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "wind_speed_10m,"
                    "weather_code"
                ),
                "timezone": "auto"
            },
            timeout=15
        )

        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data["current"]

        return (
            f"Location: {location['name']}, "
            f"{location.get('admin1', '')}, "
            f"{location['country']}\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Feels Like: {current['apparent_temperature']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind Speed: {current['wind_speed_10m']} km/h\n"
            f"Weather Code: {current['weather_code']}\n"
            f"Observation Time: {current['time']}"
        )

    except requests.RequestException as e:
        return f"Weather API request failed: {str(e)}"


from langchain_core.tools import tool
from e2b_code_interpreter import Sandbox


@tool
def code_tool(code: str) -> str:
    """
    Execute Python code in a remote E2B sandbox.
    Use this tool for calculations, data analysis,
    Python programming and testing code.
    """

    try:
        with Sandbox.create() as sandbox:

            execution = sandbox.run_code(code)

            if execution.error:
                return f"Execution Error: {execution.error}"

            output = []

            for result in execution.results:
                if result.text:
                    output.append(result.text)

            if execution.logs.stdout:
                output.extend(execution.logs.stdout)

            if execution.logs.stderr:
                output.extend(execution.logs.stderr)

            return "\n".join(output) or "Code executed successfully. No output."

    except Exception as e:
        return f"Sandbox Error: {str(e)}"


# ---------------- CREATE AGENT ----------------

agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data,code_tool],
    system_prompt="""
    You are a helpful AI assistant.
    You have access to web search and weather tools and code tools.
    Use web search for current information.
    Use the weather tool for weather-related questions.
    Always base your answers on tool results.
    """
)



# ---------------- INVOKE AGENT ----------------

print("AI Agent is ready!")

userInput = ""
while userInput.lower() != "done":
    question = input("\nAsk something: ")
    userInput = question
    response = agent.invoke(
        {
        "messages": [
            {"role": "user", "content": userInput}
        ]
        },
        config={
            "run_name": "agent_query",
            "tags": ["agent", "qa"],
            "metadata": {"source": "cli"}
        }
    )

    print("\nAI Answer:")
    print(response["messages"][-1].content)

print("Agent terminated")


