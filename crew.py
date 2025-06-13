# Install necessary libraries
!pip install crewai crewai_tools langchain_community langchain_ollama streamlit duckduckgo-search
!pip install  langchain-openai

# Import necessary modules
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os

# --- Set the OpenAI API key using environment variable ---
# It's best practice to use environment variables for sensitive information
# Replace "" with your actual API key or load it from a secure source.
os.environ["OPENAI_API_KEY"] = "**"
# Initialize LLM using the environment variable
# The api_key parameter is not strictly necessary here if OPENAI_API_KEY env var is set
llm = ChatOpenAI(
      model = "gpt-3.5-turbo",
      # api_key=os.environ["OPENAI_API_KEY"] # This is redundant if the env var is set
)

# Define the search tool
@tool
def search_web_tool(query: str):
    """
    Searches the web and returns results.
    """
    search_tool = DuckDuckGoSearchResults(num_results=10, verbose=True)
    return search_tool.run(query)

# Agents
guide_expert = Agent(
    role="City Local Guide Expert",
    goal="Provides information on things to do in the city based on user interests.",
    backstory="A local expert passionate about sharing city experiences.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=llm,
    allow_delegation=False,
)

location_expert = Agent(
    role="Travel Trip Expert",
    goal="Provides travel logistics and essential information.",
    backstory="A seasoned traveler who knows everything about different cities.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm= llm,
    allow_delegation=False,
)

planner_expert = Agent(
    role="Travel Planning Expert",
    goal="Compiles all gathered information to create a travel plan.",
    backstory="An expert in planning seamless travel itineraries.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=llm,
    allow_delegation=False,
)

# Task definitions
from_city = "India"
destination_city = "Rome"
date_from ="1 july 2025"
date_to = "7 july 2025"
interests = " sight seeing and good food"

def location_task_func(agent, from_city, destination_city, date_from, date_to):
    return Task(
        description=f"""
        In French: Provide travel-related information including accommodations, cost of living,
        visa requirements, transportation, weather, and local events.

        Traveling from: {from_city}
        Destination: {destination_city}
        Arrival Date: {date_from}
        Departure Date: {date_to}

        Respond in FRENCH if the destination is in a French-speaking country.
        """,
        expected_output="A detailed markdown report with relevant travel data.",
        agent=agent,
        output_file='city_report.md',
    )

def guide_task_func(agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        Provide a travel guide with attractions, food recommendations, and events.
        Tailor recommendations based on user interests: {interests}.

        Destination: {destination_city}
        Arrival Date: {date_from}
        Departure Date: {date_to}
        """,
        expected_output="A markdown itinerary including attractions, food, and activities.",
        agent=agent,
        output_file='guide_report.md',
    )

def planner_task_func(context, agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        Combine information into a well-structured itinerary. Include:
        - City introduction (4 paragraphs)
        - Daily travel plan with time allocations
        - Expenses and tips

        Destination: {destination_city}
        Interests: {interests}
        Arrival: {date_from}
        Departure: {date_to}
        """,
        expected_output="A structured markdown travel itinerary.",
        context=context, # Explicitly set the context here
        agent=agent,
        output_file='travel_plan.md',
    )

# Instantiate tasks using the function names to avoid overwriting
location_task_instance = location_task_func(
  location_expert,
  from_city,
  destination_city,
  date_from,
  date_to
)

guide_task_instance = guide_task_func(
  guide_expert,
  destination_city,
  interests,
  date_from,
  date_to
)

planner_task_instance = planner_task_func(
  [location_task_instance, guide_task_instance], # Pass the task instances
  planner_expert,
  destination_city,
  interests,
  date_from,
  date_to,
)

# Initialize the crew
crew = Crew(
    agents=[location_expert, guide_expert, planner_expert],
    tasks=[location_task_instance, guide_task_instance, planner_task_instance],
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    verbose=True,
    llm = llm
)

# Kick off the crew execution
result = crew.kickoff()
