# Add LCEL-based retrieval chain implementation for comparison
#Introduces create_retrieval_chain_with_lcel() using LangChain Expression
#Language to demonstrate the declarative, composable approach alongside the existing function-based implementation.

#LANGSMITH TRACING:https://smith.langchain.com/public/227e1d10-d295-471c-8d81-afe0f0e90613/r


import os
from operator import itemgetter #itemgetter is a convenient function that allows us to create a callable that retrieves an item from its operand using the provided key. In our case, we will use itemgetter to retrieve the "question" from the input dictionary when we create our LCEL chain.
# We will use itemgetter in our LCEL chain to extract the "question" from the input dictionary and pass it through the retriever and formatter to create the context for the language model. This allows us to create a more declarative and composable retrieval chain using LCEL.

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough #A simple runnable that just passes through its input. We can use this to create intermediate steps in our LCEL chain.
# We will use RunnablePassthrough to create a step in our LCEL chain that takes the input question, passes it through the retriever to get relevant documents, and then formats those documents into a context string that can be used in the prompt for the language model.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
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
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================

# Here we are not passing any arguments to the create_retrieval_chain_with_lcel function.
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

#How retrieval_chain_with_lcel works:
#1. We start by creating a retrieval chain using LCEL. The chain is defined in a declarative manner using the pipe operator (|) to chain together operations.
#2. The first part of the chain uses RunnablePassthrough.assign to create a step that takes the input question (extracted using itemgetter("question")), passes it through the retriever to get relevant documents, and then formats those documents into a context string using the format_docs function (REMEMBER - retriever returns Documents (not strings) and format_docs converts them → string). This creates a new variable called "context" that contains the formatted context string.
#3. The next part of the chain takes the formatted context and the original question and formats them into a structured prompt using the prompt_template. This creates a list of messages that can be passed to the language model.
#4. The formatted messages are then passed to the language model (llm) for generation, and the output is parsed using StrOutputParser to extract the final answer as a string.
#5. The resulting chain can be invoked with an input dictionary containing the "question" key, and it will return the generated answer based on the retrieved context.


#For example, if we invoke the chain with {"question": "What is RAG?"}, the chain will:
# Example: invoke({"question": "What is RAG?"})

# Step 1: Extract the question
# itemgetter("question") → "What is RAG?"

# Step 2: Retrieve relevant documents
# "What is RAG?" → embedding → [0.21, -0.78, 0.55, ...]
# → query Pinecone index → similarity search
# → returns:
# [
#   Document(page_content="RAG stands for Retrieval Augmented Generation..."),
#   Document(page_content="Vector databases store embeddings...")
# ]

# Step 3: Format retrieved documents into context string
# format_docs(docs) →
# "RAG stands for Retrieval Augmented Generation...\n\nVector databases store embeddings..."

# Step 4: Format prompt using context + question
# prompt_template →
# """
# Answer the question based on the context below:

# Context:
# RAG stands for Retrieval Augmented Generation...
# Vector databases store embeddings...

# Question:
# What is RAG?
# """

# Step 5: Pass prompt to LLM
# → LLM generates answer:
# "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation..."

# Step 6: Parse output
# StrOutputParser() → clean string output




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

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)

    #Here's a breakdown of the code:
    #1. We start by defining a function create_retrieval_chain_with_lcel() that creates a retrieval chain using LCEL. The chain is defined in a declarative manner using the pipe operator (|) to chain together operations.
    #2. The first part of the chain uses RunnablePassthrough.assign to create a step that takes the input question (extracted using itemgetter("question")), passes it through the retriever to get relevant documents, and then formats those documents into a context string using the format_docs function. This creates a new variable called "context" that contains the formatted context string.
    #3. The next part of the chain takes the formatted context and the original question and formats them into a structured prompt using the prompt_template. This creates a list of messages that can be passed to the language model.
    #4. The formatted messages are then passed to the language model (llm) for generation, and the output is parsed using StrOutputParser to extract the final answer as a string.
    #5. In the main block, we demonstrate the use of this LCEL-based retrieval chain by invoking it with a sample query and printing the resulting answer. We also compare it to a raw LLM invocation without RAG and a function-based retrieval chain without LCEL to highlight the advantages of using LCEL for building retrieval chains.

    #Here we have 3 implementations:
    #1. Raw LLM Invocation (No RAG): This is the simplest approach where we directly invoke the language model with the input query without any retrieval or context. This will likely produce a less accurate answer since the model has no additional information to work with.
    #2. Retrieval Chain WITHOUT LCEL: This implementation manually retrieves relevant documents, formats them into a context string, and then generates a response using the language model. This approach is more accurate than the raw invocation but is more verbose and less composable.
    #3. Retrieval Chain WITH LCEL: This implementation uses LangChain Expression Language (LCEL) to create a more declarative and composable retrieval chain. It allows for built-in streaming, async support, and better integration with LangChain's type system. This is the recommended approach for building retrieval chains in production applications.

    