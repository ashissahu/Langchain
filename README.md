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

- git add . # Stage all changes for commit
- git commit -m "Add main.py and update requirements.txt and .env files" # Commit with a message
- git push --set-upstream origin project/hello-world # Push the changes to the remote repository and set the upstream branch to project/hello-world