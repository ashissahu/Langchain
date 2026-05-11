from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv()

from langchain.agents import create_agent #create_agent is a function that creates an agent that can use tools to perform tasks.
from langchain.tools import tool #tool is a decorator that defines a function as a tool that can be used by an agent.
from langchain_core.messages import HumanMessage #HumanMessage is a class that represents a message from a human to the agent. It has a content attribute that contains the text of the message.
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch #TavilySearch is a class that provides a search tool using the Tavily API. It allows us to perform web searches and retrieve structured data in return.


#What are tools?
#Tools are functions that an agent can call to perform specific tasks. 
# They can be used to access external APIs, perform calculations, or manipulate data. 
# By using tools, agents can extend their capabilities and interact with the world in more complex ways.

#Importance of Hints, Types, and Descriptions in Tools
#Hints, types, and descriptions are crucial for tools because they provide context and guidance to the agent on how to use the tool effectively. 
# Hints can suggest when and how to use the tool, types can specify the expected input and output formats, and descriptions can explain the purpose and 
# functionality of the tool. This information helps the agent make informed decisions about when to call the tool and how to interpret its results, 
# ultimately improving the agent's performance and accuracy.


#llm = ChatOpenAI()
llm = ChatOpenAI(model="gpt-4o") #We create an instance of the ChatOpenAI model and specify that we want to use the "gpt-4o" model. 
tools = [TavilySearch()] #We also create a list of tools that includes the TavilySearch tool, which allows us to perform web searches using the Tavily API.
agent = create_agent(llm, tools=tools) #create_agent function takes an LLM and a list of tools as input and returns an agent that can use those tools to answer questions and perform tasks.


# The LLM decides which tool to call and with what arguments based on the input it receives.
# then Langchain went and ran the tool and got the result and then it passed the result back to the LLM to generate a final response for the user.

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

# In this code, we are creating an agent that can use the TavilySearch tool to perform web searches.
# We then invoke the agent with a HumanMessage that contains a query about searching for job postings for an AI engineer in Bangalore on LinkedIn. The agent will use the TavilySearch tool to perform the search and return the results, which we then print to the console.
