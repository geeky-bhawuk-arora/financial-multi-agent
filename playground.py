import os
from dotenv import load_dotenv

import phi
import phi.api
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app

# Load .env and set API key
load_dotenv()
phi.api.api_key = os.getenv("PHI_API_KEY")  

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Searches the web for the latest information and news.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tools_calls=True,
    markdown=True,
)

# Finance Agent
finance_agent = Agent(
    name="Finance AI Agent",
    role="Provides financial data like prices, analyst ratings, fundamentals, and news.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data."],
    show_tools_calls=True,
    markdown=True,
)

# Wrap both agents in a Team Agent
multi_agent = Agent(
    name="Financial Multi-Agent",
    role="Responds to financial and general info queries using a team of expert agents.",
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[finance_agent, web_search_agent],
    instructions=[
        "Always include sources.",
        "Use tables for financial data.",
    ],
    show_tools_calls=True,
    markdown=True,
)

# Create Playground with the team agent
app = Playground(multi_agent).get_app()  # Now correct

# Serve the app
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
