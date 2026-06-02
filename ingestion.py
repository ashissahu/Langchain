#uv add "protobuf<=3.20.3" #protobuf should be <=3.20.3

#I/O TASKS :- LLM API calls (OpenAI), Vector DB queries (Pinecone, Weaviate, etc.), Embedding APIs (if remote)

import asyncio # For asynchronous processing of document indexing
import os
import ssl # For configuring SSL context to use certifi certificates
from typing import Any, Dict, List
from urllib import response # For type annotations of documents and results

import certifi # For providing a bundle of trusted CA certificates for SSL connections
from dotenv import load_dotenv
from langchain_chroma import Chroma #Chroma is a local vector store that can be used to store and retrieve vector embeddings. It provides a simple interface for adding documents, generating embeddings, and performing similarity searches. In this context, it is used as an alternative to PineconeVectorStore for storing the generated embeddings of the document chunks. The persist_directory parameter specifies the directory where the Chroma database will be stored, allowing for persistent storage of the indexed documents and their embeddings.
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # For representing documents with content and metadata
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap # For crawling, extracting, and mapping documentation content

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning) # For logging messages with different colors and formats to indicate info, success, errors, and warnings

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where()) # Create an SSL context that uses the certifi certificate bundle for secure connections. This ensures that when the application makes HTTPS requests (e.g., to crawl documentation sites or connect to APIs), it can verify the server's SSL certificates against a trusted set of CAs, preventing issues with untrusted certificates and improving security.
os.environ["SSL_CERT_FILE"] = certifi.where() # Set the environment variable to point to the certifi certificate bundle, ensuring that any libraries that rely on this variable for SSL verification will use the correct certificates.
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where() # Set the environment variable for the requests library to use the certifi certificate bundle, ensuring that all HTTPS requests made using requests will be properly verified against trusted CAs.


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
) # Initialize the OpenAIEmbeddings with the specified model and parameters. This will be used to generate vector embeddings for the document chunks that will be stored in the vector store. 

#show_progress_bar=False -> Disables the progress bar during embedding generation to keep the console output clean and focused on logging messages. If set to True, it would display a progress bar in the console, which can be useful for long-running processes but may clutter the logs in this context.
#chunk_size=50 -> Sets the number of documents to process in each batch when generating embeddings. This helps manage memory usage and can improve performance by reducing the number of API calls. If set to a larger number, it may speed up the embedding generation but could also lead to higher memory consumption or longer processing times for each batch. If set to a smaller number, it may reduce memory usage but increase the total time taken due to more API calls.
#retry_min_seconds=10 -> Configures the minimum number of seconds to wait before retrying an API call if it fails due to rate limits or transient errors. This helps ensure that the application can recover gracefully from such issues without overwhelming the API with rapid retries. If set to a smaller number, it may lead to more frequent retries and potentially hit rate limits again. If set to a larger number, it may reduce the frequency of retries but could increase the overall time taken to recover from errors.


#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)
tavily_extract = TavilyExtract() # Initialize the TavilyExtract tool, which will be used to extract relevant content from the crawled documentation pages. This tool can be configured with different extraction strategies (e.g., "basic", "advanced") to control how much content is extracted from each page. In this context, it will be used to extract the main content from the documentation pages while crawling, ensuring that we get meaningful text for embedding and indexing in the vector store.
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000) # Initialize the TavilyMap tool, which will be used to control the crawling process. 

#max_depth=5 -> Sets the maximum depth of crawling, which determines how many levels of links the crawler will follow from the starting URL. A depth of 5 means that the crawler will follow links up to 5 levels deep from the initial page. This helps control the scope of crawling and prevents it from going too deep into the site, which could lead to long processing times or hitting rate limits.For example, if the starting URL is level 0, the crawler will follow links on that page (level 1), then follow links on those pages (level 2), and so on, up to level 5. Setting this to a smaller number will limit the crawl to fewer levels, while setting it to a larger number will allow for deeper crawling but may increase processing time.
#max_breadth=20 -> Sets the maximum breadth of crawling, which determines how many links the crawler will follow at each level. A breadth of 20 means that the crawler will follow up to 20 links from each page it visits. This helps control the number of pages crawled at each level and prevents the crawler from following too many links, which could lead to long processing times or hitting rate limits. For example, if a page has 50 links, the crawler will only follow 20 of them based on this setting. Setting this to a smaller number will limit the number of links followed at each level, while setting it to a larger number will allow for more links to be followed but may increase processing time.
#max_pages=1000 -> Sets the maximum number of pages to crawl in total. This helps ensure that the crawler does not exceed a certain number of pages, which can be useful for controlling processing time and avoiding hitting rate limits. For example, if the crawler reaches 1000 pages, it will stop crawling even if there are more links to follow. Setting this to a smaller number will limit the total number of pages crawled, while setting it to a larger number will allow for more pages to be crawled but may increase processing time.


# Initialize the TavilyCrawl tool, which will be used to orchestrate the crawling and extraction process. 
# This tool combines the functionality of TavilyMap and TavilyExtract to crawl the documentation site according to the specified parameters and extract relevant content from each page. 
# In this context, it will be used to crawl the documentation site, extract content, and prepare it for chunking and indexing in the vector store.
tavily_crawl = TavilyCrawl() 


# Below we are creating a coroutine function called index_documents_async that takes a list of Document objects and an optional batch size parameter. This function will be responsible for processing the documents in batches asynchronously, which can help improve performance when dealing with a large number of documents.
#Here batch_size=50 means that the function will process 50 documents at a time when generating embeddings and indexing them in the vector store. This batching approach can help manage memory usage and improve efficiency by reducing the number of API calls made to the embedding service. By processing documents in batches, we can generate embeddings for multiple documents at once, which can be more efficient than processing them one by one, especially when dealing with a large dataset.
#Coroutine means that this function can be paused and resumed, allowing for concurrent execution of multiple tasks. This is particularly useful when processing large datasets, as it can help avoid blocking the main thread and improve overall efficiency. By using asynchronous programming techniques, we can ensure that the application remains responsive while performing time-consuming operations such as generating embeddings and indexing documents in the vector store.
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    # range(0, len(documents), batch_size) creates a sequence of numbers starting from 0 up to the length of the documents list, with a step size equal to batch_size. This means that it will generate indices for the start of each batch. For example, if there are 200 documents and the batch size is 50, it will generate the indices 0, 50, 100, and 150. The list comprehension then uses these indices to create sublists (batches) of documents. Each batch will contain up to batch_size documents, and the last batch may contain fewer if the total number of documents is not perfectly divisible by the batch size.
    #for example, in first iteration, i will be 0, so documents[0:50] will create a batch of the first 50 documents. In the second iteration, i will be 50, so documents[50:100] will create a batch of the next 50 documents, and so on until all documents are processed into batches.
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    # Here we are creating a list of tasks for each batch by calling the add_batch function for each batch. 
    # The add_batch function is defined as an asynchronous function that will handle the actual addition of documents to the vector store for each batch. 
    # By using asyncio.gather(*tasks), we can run all the batch processing tasks concurrently, allowing for faster processing of the documents. 
    # This approach can significantly reduce the time it takes to index a large number of documents in the vector store compared to processing them sequentially.
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch) # This line calls the aadd_documents method of the vectorstore instance, passing in the current batch of documents. The aadd_documents method is an asynchronous function that will handle the process of generating embeddings for the documents in the batch and adding them to the vector store. By using await, we ensure that the function will pause until the embedding generation and indexing process for the current batch is complete before moving on to the next batch. This allows us to manage the flow of asynchronous tasks effectively while ensuring that each batch is processed correctly.
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    # Process batches concurrently
    # Here we are using asyncio.gather to run all the add_batch tasks concurrently. 
    # Each task corresponds to processing a batch of documents and adding them to the vector store.
    # By running these tasks concurrently, we can significantly reduce the overall time it takes to index all the documents, especially when dealing with a large number of batches. 
    # The results of each batch processing task will be collected in the results list, which can be used for further analysis or logging if needed.
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)] # This line creates a list of tasks by iterating over the batches and their corresponding indices. For each batch, it calls the add_batch function, passing in the batch of documents and its batch number (i + 1). The enumerate function is used to get both the index (i) and the batch itself from the batches list. This results in a list of asynchronous tasks that can be executed concurrently using asyncio.gather.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches): # If all batches were processed successfully, we log a success message indicating that all batches were added to the vector store without any issues. This means that the entire indexing process was completed successfully, and all documents are now available in the vector store for retrieval and use in applications such as search or question-answering.
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    # Crawl the documentation site
    # Here we invoke the TavilyCrawl tool with the specified parameters to start crawling the documentation site. 
    # The parameters include the starting URL, maximum depth, and extraction depth. 
    # The crawl will follow links according to the max_depth and max_breadth settings, and extract content based on the extract_depth setting. 
    # The results of the crawl will be stored in the variable 'res', which can then be processed further to chunk the content and index it in the vector store.
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    #all_docs = res["results"] # Extract the list of results from the crawl response. Each result represents a crawled page with its content and metadata.
    #above line for debuggung

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    #all_docs #for debugging

    # Split documents into chunks
    # #Chunking is an CPU-intensive process, especially for large documents, as it involves analyzing the text and determining appropriate split points based on the specified chunk size and overlap. 
    # SO We can not use async processing for chunking because the chunking process itself is typically CPU-bound rather than I/O-bound.
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    # Here we are calling the index_documents_async function to process the splitted documents in batches and index them in the vector store. 
    # This function will handle the generation of embeddings and the addition of documents to the vector store asynchronously, allowing for efficient processing of a large number of documents. By using await, we ensure that the main function will pause until the indexing process is complete before proceeding to log the completion of the pipeline and the summary of results.
    # batch_size=500 means that the function will process 500 documents at a time when generating embeddings and indexing them in the vector store. 
    # This batching approach can help manage memory usage and improve efficiency by reducing the number of API calls made to the embedding service. 
    # By processing documents in batches, we can generate embeddings for multiple documents at once, which can be more efficient than processing them one by one, especially when dealing with a large dataset.
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main()) # This line checks if the script is being run directly (as the main module) and if so, it executes the main function using asyncio.run. This allows us to run the asynchronous main function in an event loop, which is necessary for handling the asynchronous tasks defined in the main function and the index_documents_async function. By using asyncio.run, we can ensure that all asynchronous operations are executed properly and that the application remains responsive while performing time-consuming tasks such as crawling, extracting, chunking, and indexing documents in the vector store.


#STEP BY STEP EXPLANATION OF THE CODE:
#1. We start by importing the necessary libraries and modules for our documentation ingestion pipeline. This includes libraries for asynchronous processing, SSL configuration, environment variable management, vector store handling, text splitting, document representation, embedding generation, and crawling/extraction tools from Tavily.
#2. We load environment variables using load_dotenv(), which allows us to manage sensitive information such as API keys securely through a .env file.
#3. We configure the SSL context to use certifi certificates, ensuring that our application can securely connect to external services without encountering SSL certificate issues. This is important for making API calls to services like Tavily, which require secure connections.
#4. We initialize the OpenAIEmbeddings with the specified model and parameters, which will be used to generate vector embeddings for the document chunks that will be stored in the vector store. The parameters control aspects such as progress bar display, batch size for embedding generation, and retry behavior for API calls.
#5. We initialize the vector store (PineconeVectorStore in this case) where the generated embeddings will be stored for later retrieval and use in applications such as search or question-answering.
#6. We initialize the TavilyExtract and TavilyMap tools, which will be used to extract relevant content from the crawled documentation pages and control the crawling process, respectively. The parameters for TavilyMap help manage the depth and breadth of crawling to ensure we get meaningful content without overwhelming the crawler.
#7. We initialize the TavilyCrawl tool, which will orchestrate the crawling and extraction process by combining the functionality of TavilyMap and TavilyExtract.
#8. We define an asynchronous function index_documents_async that takes a list of Document objects and processes them in batches to generate embeddings and index them in the vector store. This function uses asynchronous programming techniques to improve efficiency when dealing with a large number of documents.
#9. In the main function, we start by logging the beginning of the documentation ingestion pipeline and then invoke the TavilyCrawl tool to start crawling the documentation site. The results of the crawl are processed to create Document objects that represent the content and metadata of each crawled page.
#10. We then use a text splitter to split the documents into smaller chunks, which can be more effectively embedded and indexed in the vector store. The parameters for the text splitter control the size of the chunks and the overlap between them.
#11. Finally, we call the index_documents_async function to process the splitted documents in batches and index them in the vector store. Once the indexing is complete, we log the completion of the pipeline and provide a summary of the results, including the number of documents extracted and chunks created.


#Flow of logic for asynchronous processing in index_documents_async:
#1. The function starts by logging the beginning of the vector storage phase and the number of documents that will be indexed.
#2. It creates batches of documents based on the specified batch size, which helps manage memory usage and improve efficiency when generating embeddings and indexing in the vector store.
#3. An inner asynchronous function add_batch is defined to handle the actual processing of each batch. This function generates embeddings for the documents in the batch and adds them to the vector store, while also logging the success or failure of each batch.
#4. The add_batch function is called for each batch of documents, and the tasks are collected in a list.
#5. The tasks are executed concurrently using asyncio.gather, which allows for faster processing of the batches. The results of each batch processing task are collected in a list.
#6. After all batches have been processed, the function counts the number of successful batches and logs a summary of the indexing results, indicating how many batches were processed successfully out of the total number of batches. This provides insight into the overall success of the indexing process and helps identify any issues that may have occurred during batch processing.


#⚙️ 1. async → Define an asynchronous function
#####################################################
#Use async when you are writing a function that:
  # Will perform I/O operations
  # Needs to be run concurrently
  #ex - async def fetch_data():
        #  ...
#👉 Think: “This function can pause and resume later”

#⏸️ 2. await → Wait for an async operation
#################################################
#Use await inside an async function when:
    #Calling another async function
    #Waiting for I/O to complete

    #async def fetch_data():
        #response = await api_call()
        #return response

#👉 Think: “Pause here until result comes”


#🚀 3. asyncio → Run and manage async tasks
########################################
# asyncio is the engine / event loop manager
# 🔹 Run your program

# if __name__ == "__main__":
#     asyncio.run(main())

#👉 Starts the async system

#🔹 Run tasks concurrently
# tasks = [task1(), task2(), task3()]
# results = await asyncio.gather(*tasks)

#👉 Run multiple tasks at once(in parallel) and wait for all to finish



#🔥 Key Differences (Quick Table)
#Keyword	Purpose
#async	Define async function
#await	Wait for async result
#asyncio	Run/manage async tasks
#gather()	Run multiple tasks concurrently


#🧠 Mental Model
#async → “I can pause”
#await → “Pause here”
#asyncio → “Orchestrator”

