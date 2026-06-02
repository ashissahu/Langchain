#Langsmith Tracing - https://smith.langchain.com/public/53c508cf-e627-4f0d-ab86-9dd9e519e2c6/r
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2026", embedding=embeddings
)
#Does the index nme need to be the same as the one used in ingestion? 
# Yes, it should match the index name used during ingestion to ensure that we are retrieving from the correct vector store where the document embeddings were indexed. 
# If the index name does not match, the retrieval tool will not be able to find the relevant documents, and the RAG pipeline will not work as intended. 
# So make sure to use the same index

# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")

#In the below toolfunction,in response_format, we specify "content_and_artifact" to indicate that the tool will return both a content string (the serialized retrieved documents) and an artifact (the raw list of Document objects). 
# This allows the agent to access both the human-readable context for generating answers and the original documents for reference if needed. 
# The agent can then use the retrieved context to generate more informed answers to user queries about LangChain documentation.
#If response_format is not specified, the tool will default to returning just the content string, and the raw documents would not be accessible to the agent, which could limit its ability to provide detailed answers or cite sources effectively.
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4) #invoke will call the retriever and return the top k most similar documents based on the query embeddings. The retrieved_docs variable will contain a list of Document objects that are relevant to the user's query.
    
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return both serialized content and raw documents
    return serialized, retrieved_docs #We will send the serialized string to the model(LLM) for context, and also return the raw Document objects as an artifact that the agent can access if needed for more detailed information or citation.


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    #Here we are only considering last message as answer here because the agent will generate a response after processing the retrieved context, and that response will be in the last message of the conversation.
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    } #here we return a dictionary containing the generated answer by LLM and the list of context documents that were retrieved during the agent's processing. This allows us to see not only the final answer but also the sources that were used to generate that answer.

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)

#STEP BY STEP EXPLANATION OF THE CODE:
#1. We start by importing necessary libraries and modules, including os for environment variable management, typing for type annotations, dotenv for loading environment variables from a .env file, and various components from the langchain library for building the RAG pipeline.
#2. We load environment variables using load_dotenv(), which allows us to access API keys and other configuration settings stored in a .env file.
#3. We initialize the OpenAIEmbeddings with the specified model, which will be used to generate embeddings for the documents in our vector store.
#4. We set up the PineconeVectorStore, specifying the index name and the embeddings instance we just created. This vector store will be used to store and retrieve document embeddings.
#5. We initialize the chat model using init_chat_model, specifying the model name and provider. This model will be used to generate responses based on the retrieved context.
#6. We define a tool function retrieve_context that takes a query as input and retrieves relevant documentation from the vector store. It retrieves the top 4 most similar documents, serializes them into a string format, and returns both the serialized content and the raw documents.
#7. We define the main function run_llm that takes a user query as input and runs the RAG pipeline. It creates an agent with the retrieval tool, constructs the system prompt, builds the messages list, and invokes the agent. The function then extracts the generated answer from the response and collects any context documents that were retrieved during the process. Finally, it returns a dictionary containing both the answer and the context documents.
#8. In the main block, we call the run_llm function with a sample query about "deep agents" and print the result, which includes the generated answer and the context documents that were retrieved to support the answer.

# This code sets up a simple RAG pipeline using LangChain, where a user query is processed by an agent that retrieves relevant documentation from a vector store and generates an answer based on that context. The retrieved documents are also included in the output for reference.