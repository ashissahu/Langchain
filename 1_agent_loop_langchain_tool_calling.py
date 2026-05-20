#Using OPENAI
#Langsmith Tracing -https://smith.langchain.com/public/889f17de-3132-480a-9b3e-4d34b70678ee/r

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model #init_chat_model is a function that initializes a chat model based on the environment variables. It checks for the presence of certain environment variables to determine which chat model to use. For example, if the OPENAI_API_KEY environment variable is set, it will initialize a ChatOpenAI model. If the AZURE_OPENAI_API_KEY environment variable is set, it will initialize a ChatAzureOpenAI model. If neither of these environment variables are set, it will raise an error.
from langchain.tools import tool #tool is a decorator that defines a function as a tool that can be used by an agent.
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage #These are classes that represent different types of messages that can be sent to the agent. HumanMessage represents a message from the user, SystemMessage represents a message from the system, and ToolMessage represents a message from a tool.
from langsmith import traceable #traceable is a decorator that allows us to trace the execution of a function and log it to Langsmith, which is a tool for monitoring and analyzing the performance of language models.

MAX_ITERATIONS = 10 #This constant defines the maximum number of iterations that the agent will perform when trying to generate a response. It is used to prevent infinite loops and ensure that the agent does not get stuck in a cycle of generating responses without making progress towards a final answer.
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---
#Here we define two tools that the agent can use: get_product_price and apply_discount. 
# These tools are decorated with the @tool decorator, which allows them to be called by the agent during its reasoning process. 
# The get_product_price tool looks up the price of a product in a predefined catalog, 
# while the apply_discount tool applies a discount tier to a given price and returns the final price after the discount is applied.

# Here we are providing type hints for the tool functions, which can help the language model understand what types of inputs and outputs to expect when calling these tools. The get_product_price function takes a string input (the name of the product) and returns a float (the price of the product). 
# The apply_discount function takes a float input (the price) and a string input (the discount tier) and returns a float (the final price after applying the discount).
# We are also providing docstrings for each tool function, which describe what the function does and what its inputs and outputs are. This can help the language model understand how to use these tools correctly when it decides to call them during its reasoning process.

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop") #This decorator allows us to trace the execution of the run_agent function and log it to Langsmith. By using this decorator, we can see how the agent is making decisions, which tools it is calling, and how it is generating its responses. This can help us understand the agent's behavior and identify any issues or areas for improvement.
def run_agent(question: str):
    tools = [get_product_price, apply_discount] #List of tools we want to make available to the agent. These are the functions that we defined earlier with the @tool decorator. The agent can call these tools during its reasoning process to get information or perform actions that will help it generate a response to the user's question.
    tools_dict = {t.name: t for t in tools} #This creates a dictionary that maps the name of each tool to the tool function itself. This allows us to easily look up a tool by its name when the agent decides to call it. The agent will specify the name of the tool it wants to call, and we can use this dictionary to find the corresponding function and execute it with the provided arguments.
    print(f"tools_dict - {tools_dict}")

    #llm = init_chat_model(f"openai:gpt-5.2", temperature=0) #This initializes a chat model using the init_chat_model function. The model is specified as "openai:{MODEL}", which indicates that we want to use the specified OpenAI model. The temperature parameter is set to 0, which means that the model will generate deterministic responses (i.e., it will always generate the same response for the same input). This is useful for testing and debugging, as it allows us to see consistent behavior from the model.
    #not working properly with gpt -5.2
    llm = init_chat_model(f"openai:gpt-5", temperature=0) #Using GPT-5 instead of GPT-5.2 because the latter is not working properly with tool calling in this example. The GPT-5 model supports function calling, which allows us to use the bind_tools method to associate the tools we defined with the language model. This way, the model can call these tools during its reasoning process to generate more informed responses based on the information provided by the tools.
    llm_with_tools = llm.bind_tools(tools) # binds the tools to llm.Only works if llm supports function calling . This allows the model to call the tools we defined earlier during its reasoning process. When the model generates a response, it can include instructions to call a specific tool with certain arguments. The bind_tools method enables this functionality by associating the tools with the language model, allowing it to use them as part of its response generation.

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ] #List of messages we will send to llm

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages) #This sends the list of messages to the language model and gets a response back. The invoke method processes the messages and generates a response based on the input. The response can include content (text) as well as tool calls (instructions to use tools). The ai_message variable will contain the model's response, which we can then analyze to see if it includes any tool calls or if it is the final answer to the user's question.

        tool_calls = ai_message.tool_calls #Which tools the model decided should be invoked. This works only if:1.You passed tools/functions to the model 2.Your model supports tool calling (e.g. GPT-4, some Ollama models with function calling) . 
        #The tool_calls variable will contain a list of tool calls that the model has decided to make based on its response. 
        # Each tool call will include the name of the tool to be called, the arguments to be passed to the tool, 
        # and a unique identifier for the tool call. 
        # We can use this information to execute the appropriate tools and provide the results back to the model in subsequent iterations.

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0] #We are only processing the first tool call in the list of tool calls. This means that if the model decides to call multiple tools in its response, we will only execute the first one and ignore the rest for that iteration. This is done to simplify the processing and ensure that we are only handling one tool call at a time. The tool_call variable will contain the information about the first tool call, including the name of the tool, the arguments to be passed to the tool, and the unique identifier for that tool call. We can then use this information to execute the appropriate tool and provide the results back to the model in subsequent iterations.
        tool_name = tool_call.get("name") #The name of the tool to call. This is the name that the model specified in its response when it decided to call a tool. We will use this name to look up the corresponding tool function in our tools_dict and execute it with the provided arguments.
        tool_args = tool_call.get("args", {}) #The arguments to pass to the tool. This is a dictionary of arguments that the model specified in its response when it decided to call a tool. We will pass these arguments to the tool function when we execute it. If there are no arguments provided, we will use an empty dictionary as the default value.
        tool_call_id = tool_call.get("id") #A unique identifier for this tool call. This ID can be used to track the tool call and associate the results back to it when we provide the results to the model in subsequent iterations. It helps us keep track of which tool call corresponds to which result, especially if there are multiple tool calls being made across different iterations.

        print(f"  [Tool Call] {tool_call} [Tool Selected] {tool_name} with args: {tool_args} and tool_call_id: {tool_call_id}")

        tool_to_use = tools_dict.get(tool_name) #Look up the tool function based on the tool name provided by the model. We use the tools_dict that we created earlier to find the corresponding tool function for the given tool name. If the tool name is not found in the dictionary, it means that the model has requested a tool that we do not have available, and we will raise an error in that case.
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args) #This executes the tool function with the provided arguments and gets the result back. The invoke method of the tool function takes the arguments specified by the model and runs the tool's logic to produce an output. The observation variable will contain the result of executing the tool, which we can then provide back to the model in subsequent iterations to help it generate a more informed response.

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message) #store the model’s response, which may contain: content (text) ,tool_calls (instructions to use tools). 
        #This allows us to keep a history of the conversation and the model's reasoning process. The messages list will now include the model's response, which we can use in subsequent iterations to provide context for the model's next response.
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        ) #It tells the model:“You asked to call a tool — here is the result.” 
        #The content of the ToolMessage is the result of executing the tool, and the tool_call_id is used to associate this result with the specific tool call that the model made in its response. By appending this ToolMessage to the messages list, we provide the model with the information it needs to continue its reasoning process in the next iteration, allowing it to use the results of the tool calls to generate a more informed response.

    print("ERROR: Max iterations reached without a final answer") #
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?") #This is the query we are asking the agent to answer. The agent will use the tools we defined (get_product_price and apply_discount) to find the price of a laptop and then apply the gold discount to it. The agent will follow the rules we specified in the SystemMessage to ensure that it calls the tools in the correct order and does not make any assumptions about the prices or discounts. The final result will be printed at the end, showing the price of the laptop after applying the gold discount.

# In this example, we have a simple agent that can answer questions about product prices and discounts using 
# two tools: get_product_price and apply_discount. 
# The agent follows strict rules to ensure that it uses the tools correctly and does not make any assumptions. 
# By using the @tool decorator and the bind_tools method, we can easily integrate these tools into the agent's reasoning process, 
# allowing it to generate accurate responses based on the information provided by the tools.