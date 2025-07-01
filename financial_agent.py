from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",  
    role="Searches the web for information and news.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tools_calls=True,
    markdown=True,
)

# Finance Agent
finance_agent = Agent(
    name="Finance AI Agent",
    role="Provides financial data, stock prices, and analyst recommendations.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Use tables to display the data."],
    show_tools_calls=True,
    markdown=True,
)

# Multi-Agent Setup
multi_ai_agent = Agent( 
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_search_agent, finance_agent],
    instructions=[
        "Always include sources.",
        "Always use tables to display the data."
    ],
    show_tools_calls=True,
    markdown=True,
)

response = multi_ai_agent.run("Summarize analyst recommendations and share the latest news for NVDA.")
print(response)
