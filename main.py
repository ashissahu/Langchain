#Langsmith Tracing - https://smith.langchain.com/o/a7f8bd85-b797-5733-8038-ce1ba498c5e8/projects/p/c710d3bf-1908-4cd1-ab61-866062ae04c7?columnVisibilityModel_runs%3AcolumnVisibilityModel%3Adefault=%7B%22feedback_stats%22%3Afalse%2C%22reference_example%22%3Afalse%7D&scroll_to=output&peek=20260511T132030Z019e1732-4333-7090-8e42-92d66058c4f0&peeked_trace=20260511T132025966104Z019e1732-316e-76a0-82a9-6cdc4f502ae4

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

# In this code, we define a main function that prints a greeting message and then invokes the agent with a HumanMessage containing a 
# query about searching for job postings for an AI engineer in Bangalore on LinkedIn. 
# The agent will use the TavilySearch tool to perform the search and generate a response based on the query. 
# Finally, we print the result of the agent's response with response_format=AgentResponse, which will include the answer and the sources used to generate that answer.
