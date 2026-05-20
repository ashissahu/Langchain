from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily=TavilyClient()

@tool
def search(query:str) -> str:
    """
    Tool that Searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-4o")
tools = [search]
agent = create_agent(model=llm, tools=tools)



def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages":HumanMessage(
                content="search for 10 job postings for an ai engineer using langchain in  Bangalore  on linkedin and list their details?"
                )
        }
    )
    print(result)


if __name__ == "__main__":
    main()

#Here's a breakdown of the code:
#1. We import necessary libraries and load environment variables.
#2. We define a tool function search that takes a query string and returns search results using the TavilyClient.
#3. We initialize a ChatOpenAI LLM and create an agent with the LLM and the search tool.
#4. In the main function, when  we invoke the agent with a HumanMessage containing a search query about job postings for an AI engineer in Bangalore on LinkedIn, The agent will process this message, decide to use the search tool, and return the results.
#5. Finally, we print the result returned by the agent.
