import google.generativeai as genai
import wikipedia
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType


# Set up Gemini API key
API_KEY = "you api key"
genai.configure(api_key=API_KEY)

# Initialize Gemini LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=API_KEY)

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia error: {e}"

# Define Wikipedia search tool
wikipedia_tool = Tool(
    name="Wikipedia",
    func=search_wikipedia,
    description="Search Wikipedia and get a short summary of a topic."
)

# Initialize agent with tools
agent = initialize_agent(
    tools=[wikipedia_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Function to run the agent
def wikipedia_agent(query):
    return agent.run(query)

# Test the agent
if __name__ == "__main__":
    query = "Albert Einstein"
    print(f"Query: {query}\nResponse: {wikipedia_agent(query)}")
