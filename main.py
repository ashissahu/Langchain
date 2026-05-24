#LANGSMITH TRACING - https://smith.langchain.com/public/9e968a21-f6f0-4663-9570-1c64c0b722b3/r
#LANGSMITH TRACING - https://smith.langchain.com/o/a7f8bd85-b797-5733-8038-ce1ba498c5e8/projects/p/39de7e60-5367-4a74-a12b-71d45256eaac?runview=traces&peek=20260524T113453Z019e59c4-3d79-7fb3-9cb0-1ae423c01044&peeked_trace=20260524T113453561943Z019e59c4-3d79-7fb3-9cb0-1ae423c01044&columnVisibilityModel_runs%3AcolumnVisibilityModel%3Adefault=%7B%22feedback_stats%22%3Afalse%2C%22reference_example%22%3Afalse%7D&scroll_to=feedback

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings) #This will connect to the existing Pinecone index that we created during the ingestion process. We are not creating a new index here, but rather connecting to the existing one to perform retrieval operations.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #This will create a retriever that can be used to retrieve relevant documents from the Pinecone index based on a query. The search_kwargs parameter allows us to specify additional parameters for the retrieval process, such as the number of top results to return (k=3 in this case).

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query) #This will use the retriever to retrieve relevant documents from the Pinecone index based on the input query. The retrieved documents will be returned as a list of document objects, which we can then format and use as context for generating a response.

    # Step 2: Format documents into context string
    context = format_docs(docs) #This will take the list of retrieved document objects and format them into a single string that can be used as context for the language model. The format_docs function is a helper function that concatenates the content of each retrieved document into a single string, separated by double newlines ("\n\n") for better readability. This formatted context will then be passed to the prompt template for generating a response based on the retrieved information.

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query) #This will take the formatted context string and the original query and format them into a structured prompt that can be passed to the language model for generation. The prompt_template is a ChatPromptTemplate that defines the structure of the prompt, including placeholders for the context and question. The format_messages method will replace these placeholders with the actual context and question values, resulting in a list of messages that can be used as input for the language model to generate a response.

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages) #This will take the formatted messages (which include the context and question) and pass them to the language model (LLM) for generation. The invoke method will process the input messages and generate a response based on the provided context and question. The generated response will be returned as an object that contains the content of the response, which we can then extract and return as the final answer to the query.

    # Step 5: Return the content
    return response.content


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

#Here's a breakdown of the code:
#1. We start by importing the necessary libraries and loading environment variables using dotenv.
#2. We initialize the components needed for our RAG system, including the OpenAI embeddings, the ChatOpenAI language model, and the Pinecone vector store retriever. The vector store is connected to an existing Pinecone index that we created during the ingestion process.
#3. We define a prompt template using ChatPromptTemplate, which specifies how the retrieved context and the user's question will be structured when passed to the language model for generation.
#4. We define a helper function format_docs to format the retrieved documents into a single string that can be used as context for the language model.
#5. We implement a retrieval chain function retrieval_chain_without_lcel that performs the retrieval and generation steps manually without using LCEL. This function retrieves relevant documents based on the input query, formats them into context, formats the prompt, and invokes the language model to generate a response.
#6. In the main block, we define a query and demonstrate two implementations: a raw invocation of the language model without RAG, and the retrieval chain implementation without LCEL. We print the results for both implementations to compare the answers generated by the language model with and without the use of retrieved context.

#Here user query is converted into an embedding and then used to query the Pinecone index for similar embeddings. The retrieved documents are then formatted and passed as context to the language model to generate a response that is grounded in the retrieved information. This allows us to provide more accurate and informed answers to user queries by leveraging the relevant information stored in the Pinecone index.
# User query embedding is happening implicitly within the retriever.invoke(query) call, where the retriever uses the query to retrieve relevant documents based on vector similarity in the embedding space. The retrieved documents are then used as context for generating a response from the language model.


#The main disadvantages of the implementation without LCEL are:
#1. Manual step-by-step execution: The retrieval and generation steps are executed manually, which can be more error-prone and less efficient compared to using a structured chain that handles these steps automatically.
#2. No built-in streaming support: This implementation does not have built-in support for streaming responses from the language model, which can be a limitation for applications that require real-time or incremental responses.
#3. No async support without additional code: This implementation does not natively support asynchronous execution, which can be a drawback for applications that need to handle multiple requests concurrently or require non-blocking operations.
#4. Harder to compose with other chains: This implementation is less modular and harder to compose with other chains or components in a larger system, as it does not follow a structured approach to chaining operations together.
#5. More verbose and error-prone: The manual handling of each step can lead to more verbose code and a higher likelihood of errors, especially as the complexity of the retrieval and generation process increases. Using a structured chain with LCEL can help mitigate these issues by providing a more organized and efficient way to manage the flow of data and operations.