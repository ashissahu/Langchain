#Langsmith Tracing - https://smith.langchain.com/public/d4c38e15-f911-4a83-a8ee-7269270f82df/r

from typing import List #List is a type hint that indicates that a variable is expected to be a list of a certain type. For example, List[int] indicates that the variable should be a list of integers. This helps with code readability and can also assist with static type checking.
from pydantic import BaseModel, Field #BaseModel is a class from the Pydantic library that provides data validation and parsing. It allows us to define data models with specific fields and types, and it will automatically validate the input data against those definitions. Field is a function that allows us to specify additional metadata for each field in the model, such as default values, descriptions, and validation rules.

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent #create_agent is a function that creates an agent that can use tools to perform tasks.
from langchain.tools import tool #tool is a decorator that defines a function as a tool that can be used by an agent.
from langchain_core.messages import HumanMessage #HumanMessage is a class that represents a message from a human to the agent. It has a content attribute that contains the text of the message.
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch #TavilySearch is a class that provides a search tool using the Tavily API. It allows us to perform web searches and retrieve structured data in return.

class Source(BaseModel): #Here we define a Pydantic model called Source, which has a single field called url. This model will be used to represent the sources that the agent uses to generate its answers. The url field is a string that contains the URL of the source, and it has a description that explains its purpose.
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel): #Here we define another Pydantic model called AgentResponse, which has two fields: answer and sources. The answer field is a string that contains the agent's answer to the query, and it has a description that explains its purpose. The sources field is a list of Source objects, which represent the sources used by the agent to generate the answer. It has a default factory that creates an empty list if no sources are provided, and it also has a description that explains its purpose.
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer" 
    )

#llm = ChatOpenAI()
llm = ChatOpenAI(model="gpt-4o") #We create an instance of the ChatOpenAI model and specify that we want to use the "gpt-4o" model. 
tools = [TavilySearch()] #We also create a list of tools that includes the TavilySearch tool, which allows us to perform web searches using the Tavily API.
agent = create_agent(llm, tools=tools, response_format=AgentResponse) #We then create an agent using the create_agent function, passing in the language model (llm), the list of tools, and the response format (AgentResponse).
#response_format=AgentResponse indicates that we want the agent to return responses in the format defined by the AgentResponse model, which includes an answer and a list of sources.

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

#Here's a breakdown of the code:
#1. We import necessary libraries and load environment variables.
#2. We define two Pydantic models, Source and AgentResponse, to structure the data for sources and agent responses.
#3. We create an instance of the ChatOpenAI model and specify the "gpt-4o" model to use.
#4. We create a list of tools that includes the TavilySearch tool for web searching.
#5. We create an agent using the create_agent function, passing in the language model, the tools, and the response format.
#6. We define a main function that invokes the agent with a HumanMessage containing a query about job postings for an AI engineer in Bangalore on LinkedIn.
#7. Finally, we call the main function when the script is executed. The agent will process the query, decide which tool to use, execute the tool, and return a structured response containing the answer and the sources used.
