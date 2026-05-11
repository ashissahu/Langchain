#LANGSMITH TRACING -https://smith.langchain.com/o/a7f8bd85-b797-5733-8038-ce1ba498c5e8/projects/p/c710d3bf-1908-4cd1-ab61-866062ae04c7?columnVisibilityModel_runs%3AcolumnVisibilityModel%3Adefault=%7B%22feedback_stats%22%3Afalse%2C%22reference_example%22%3Afalse%7D&peek=20260511T105010Z05c64f08-d8c1-47bc-b344-38ed26d93a3a&peeked_trace=20260511T105008690853Zff8874da-dc5f-4692-a13d-f754c8a7e51c&scroll_to=metadata

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent #create_agent is a function that creates an agent that can use tools to perform tasks.
from langchain.tools import tool #tool is a decorator that defines a function as a tool that can be used by an agent.
from langchain_core.messages import HumanMessage #HumanMessage is a class that represents a message from a human to the agent. It has a content attribute that contains the text of the message.
from langchain_openai import ChatOpenAI

#What are tools?
#Tools are functions that an agent can call to perform specific tasks. 
# They can be used to access external APIs, perform calculations, or manipulate data. 
# By using tools, agents can extend their capabilities and interact with the world in more complex ways.

#Importance of Hints, Types, and Descriptions in Tools
#Hints, types, and descriptions are crucial for tools because they provide context and guidance to the agent on how to use the tool effectively. 
# Hints can suggest when and how to use the tool, types can specify the expected input and output formats, and descriptions can explain the purpose and 
# functionality of the tool. This information helps the agent make informed decisions about when to call the tool and how to interpret its results, 
# ultimately improving the agent's performance and accuracy.

@tool
def search(query: str) -> str:
    """Tool that searches the web for the given query and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for '{query}'...")
    return "Indian Weather is hot"

#llm = ChatOpenAI()
llm = ChatOpenAI(model="gpt-4o")
tools = [search]
agent = create_agent(llm, tools=tools) #create_agent function takes an LLM and a list of tools as input and returns an agent that can use those tools to answer questions and perform tasks.

#If we see the function variables in the debug  output , we can see that the agent has a variable called "messages" which is a list of HumanMessage, AIMessage, ToolMessage, and AIMessage.
# When we call the agent with a HumanMessage, the agent will process the message and decide which tool to call based on the content of the message.
# For example, if the message is "What is the weather in India today?", the agent will recognize that it needs to call the search tool with the query "What is the weather in India today?".
# The agent will then call the search tool, get the result, and pass the result back to the LLM to generate a final response for the user.


# The LLM decides which tool to call and with what arguments based on the input it receives.
# then Langchain went and ran the tool and got the result and then it passed the result back to the LLM to generate a final response for the user.

def main():
    print("Hello from langchain!")
    response = agent.invoke({"messages": [HumanMessage(content="What is the weather in India?")]})
    print(response)


if __name__ == "__main__":
    main()

## In this code, we have defined a tool called "search" that simulates searching the web for a given query and returns a hardcoded result.
# We then create an agent using the ChatOpenAI LLM and the search tool. Finally, we invoke the agent with a HumanMessage asking about the weather in India today, and print the response.
