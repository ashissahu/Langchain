# Langchain

uv is an extremely fast Python package and project manager (written in Rust) that serves as a modern replacement for pip, pip-tools, and virtualenv.

- uv init . # Create a virtual environment in the current directory
- uv venv # create a virtual environment using uv
- .venv\Scripts\activate # Activate the virtual environment (Windows)
- uv pip list # List installed packages in the virtual environment
- uv add langchain-openai openai python-dotenv pydantic streamlit # Add required packages to the virtual environment
- uv pip install -r requirements.txt # Install dependencies from the requirements file
- uv run app.py #Runs your script using the project's environment.
- uv run streamlit run app.py # Runs your Streamlit app using the project's environment.


- git checkout --orphan project/hello-world # Create a new orphan branch named "project/hello-world"
- git rm -rf . # Remove all files from the index
- echo "langchain" > requirements.txt # Create a new requirements.txt file with the content "langchain"
- git add requirements.txt # Add the new requirements.txt file to the index
- git commit -m "Add requirements.txt with langchain" # Commit the changes with a message
- git checkout main # Switch back to the main branch