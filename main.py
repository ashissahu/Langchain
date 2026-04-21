from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
import os


def main():
    print("Hello from langchain!")
    # print("Your OpenAI API key is:", os.getenv("OPENAI_API_KEY"))
    # print("Your Gemini API key is:", os.getenv("GEMINI_API_KEY"))
    # print("Your Tavily API key is:", os.getenv("TAVILY_API_KEY"))
    # print("Your Groq API key is:", os.getenv("GROQ_API_KEY"))
    # print("Your Langchain API key is:", os.getenv("LANGCHAIN_API_KEY"))


if __name__ == "__main__":
    main()
