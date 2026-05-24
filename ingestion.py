#https://medium.com/@EjiroOnose/vector-database-what-is-it-and-why-you-should-know-it-ae7e7dca82a4
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone




if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("D:\Langchain_Learn\mediumblog1.txt",encoding="utf-8") #encoding="utf-8",  (or)  autodetect_encoding=True
    document = loader.load() #This will return a list of documents, even if there is only one document in the file. Each document is represented as a dictionary with a "page_content" key containing the text content of the document.
    print(f"loaded {len(document)} document(s)")
    #print(document)
    #print(f"document content: {document[0].page_content[100]}...") #To see the first 100 characters of the document content

    #print("*"*50)
    #print("*"*50)
    #print("*"*50)

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # Default slitting character is "\n\n" (double newline), but you can specify any character you want to split on. In this case, we are splitting on every 1000 characters without any overlap.
    # Some chunk sizes are more than 1000 characters, but that is because the default splitting character is "\n\n" (double newline), and some chunks may contain more than 1000 characters if there are no double newlines within that range.
    # \n\n is used to split the text into paragraphs, so if there are long paragraphs without double newlines, they may exceed the specified chunk size.
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    #To see size of each chunk
    #for i, text in enumerate(texts):
        #print(f"Chunk {i+1} size: {len(text.page_content)} characters")
        #print(f"Chunk {i+1} content: {text.page_content}...") #To see the first 100 characters of each chunk
        #print("*"*50)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")) #Default embedding model is "text-embedding-ada-002", but you can specify any embedding model you want to use by passing the model name as a parameter to the OpenAIEmbeddings constructor.  
    #To see the embedding model being used
    print(f"Using embedding model: {embeddings.model}")

    print("ingesting...")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # Check existing indexes
    print(f"Existing indexes: {pc.list_indexes().names()}")

    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")

    #Here's a breakdown of the code:
    #1. We start by importing the necessary libraries and loading environment variables using dotenv.
    #2. We use the TextLoader from LangChain to load a text document from a specified file path. The loaded document is returned as a list of dictionaries, where each dictionary contains the text content under the "page_content" key.
    #3. We then use the CharacterTextSplitter to split the loaded document into smaller chunks of text. The chunk size is set to 1000 characters, and there is no overlap between chunks. The resulting chunks are stored in the "texts" variable.
    #4. Next, we create an instance of OpenAIEmbeddings to generate embeddings for  the text chunks. The OpenAI API key is passed as an argument to the constructor. We also print the embedding model being used.
    #5. Finally, we use the PineconeVectorStore to store the generated embeddings in a Pinecone index. We create the vector store from the documents (text chunks) and the embeddings, specifying the index name from the environment variables. After the ingestion process is complete, we print a finish message.

    # We can see the embedding details in the Pinecone dashboard, where we can view the stored vectors and their associated metadata. This allows us to efficiently retrieve relevant information based on vector similarity when we later query the index for specific information.

