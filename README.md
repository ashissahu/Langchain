# Langchain
 - Langchain Docs:Agents ://docs.langchain.com/oss/python/langchain/agents
 - TAVILY : https://app.tavily.com/home?utm_campaign=eden_marco&utm_medium=socials&utm_source=linkedin
          : https://docs.tavily.com/sdk/python/quick-start
 - Langchain Tavily Integration : https://docs.langchain.com/oss/python/integrations/providers/tavily
 
- Structured Output - Pydantic - https://docs.langchain.com/oss/python/langchain/structured-output

- ChatPromptTemplate Docs - https://reference.langchain.com/python/langchain-core/prompts
- ChatModels Docs - https://reference.langchain.com/python/langchain/models

# LangChain Workflow — One-Line Pointers
- User Query → Raw input from the user, often unstructured and ambiguous.
- Prompt Template → Transforms the query into a structured, instruction-rich prompt.
- Language Model (1st Call) → Interprets the prompt and generates intermediate structured output.
- Output Parser → Converts LLM output into clean, validated, machine-readable data.
- External API / Tool Call → Fetches real-world data or executes logic using parsed inputs.
- Final LLM Call → Synthesizes API results into a meaningful, human-readable response.
- Final Output → Delivers the refined, accurate answer back to the user.


# Using OLLAMA
- https://ollama.com/search

- in command prompt > ollama       #in terminal, run the following commands to set up Ollama and pull the Gemma 4 model:
- ollama pull qwen3:1.7b               # Pull the qwen3 model from Ollama
- ollama list                      # List the available models in Ollama
- ollama rm <MODEL_NAME>           #Run the following command replacing MODEL_NAME with the name from your list to Remove the model



uv is an extremely fast Python package and project manager (written in Rust) that serves as a modern replacement for pip, pip-tools, and virtualenv.

- # delete old env - rm -rf .venv   # (Windows: rmdir /s /q .venv)

- uv init . # Create a virtual environment in the current directory
- uv venv # create a virtual environment using uv. It will install latest python version
- uv venv --python 3.10 #install specific version of pythin in environment
- .venv\Scripts\activate # Activate the virtual environment (Windows)
- uv pip list # List installed packages in the virtual environment
- uv add langchain-openai openai python-dotenv pydantic streamlit # Add required packages to the virtual environment
- uv pip install -r requirements.txt # Install dependencies from the requirements file
- uv run app.py #Runs your script using the project's environment.
- uv run streamlit run app.py # Runs your Streamlit app using the project's environment.


- git checkout --orphan project/hello-world # Create a new orphan branch named "project/hello-world"
- git rm -rf . # Remove all files from the index

- git add . # Stage all changes for commit
- git commit -m "Add main.py and update requirements.txt and .env files" # Commit with a message
- git push --set-upstream origin project/hello-world # Push the changes to the remote repository and set the upstream branch to project/hello-world
- git checkout -b project/agents-under_the_hood commitid  #This command creates a new branch called "project/agents-under_the_hood" and checks it out. The "commitid" is a placeholder for the specific commit ID that you want to base this new branch on. You would replace "commitid" with the actual commit hash that you want to use as the starting point for your new branch. This allows you to work on a new feature or project while keeping the main branch clean and organized.



