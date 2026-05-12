#Using OPENAI
#Langsmith Tracing -https://smith.langchain.com/public/951afb3e-d76f-4674-9f61-9f4f8dbbc449/r

#Here we are using Raw function calling rather than implementing langchain functionality directly

from dotenv import load_dotenv
load_dotenv()

import ollama #Ollama is a library that provides an interface to interact with Ollama models. It allows us to initialize and use language models hosted on Ollama's platform. In this code, we are using Ollama to initialize a chat model that supports function calling, which is essential for the agent to be able to call the tools we defined during its reasoning process.
from langsmith import traceable #traceable is a decorator that allows us to trace the execution of a function and log it to Langsmith, which is a tool for monitoring and analyzing the performance of language models.

MAX_ITERATIONS = 10 #This constant defines the maximum number of iterations that the agent will perform when trying to generate a response. It is used to prevent infinite loops and ensure that the agent does not get stuck in a cycle of generating responses without making progress towards a final answer.
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---
@traceable(run_type="tool") #This decorator allows us to trace the execution of the get_product_price function and log it to Langsmith. By using this decorator, we can see when the function is called, what arguments are passed to it, and what value it returns. This can help us understand how the agent is using this tool and identify any issues or areas for improvement in its usage.
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool") #This decorator allows us to trace the execution of the apply_discount function and log it to Langsmith. By using this decorator, we can see when the function is called, what arguments are passed to it, and what value it returns. This can help us understand how the agent is using this tool and identify any issues or areas for improvement in its usage.
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

#run_type="tool" in the @traceable decorator indicates that these functions are tools that the agent can call during its reasoning process. By marking them as tools, we can track their usage and see how the agent is utilizing them to generate responses. This is important for understanding the agent's behavior and ensuring that it is using the tools correctly to arrive at accurate answers for the user's questions.
#It is entirrely for traceability and has no effect on the actual execution of the functions. The functions will work the same way regardless of whether we use the @traceable decorator or not. The decorator is simply a way to log the execution of these functions to Langsmith for monitoring and analysis purposes.

#These 2 are still python functions and we need to convert them into tools that the language model can call during its reasoning process. 
# To make these functions available as tools to the model, we will need to define a JSON schema for each function that describes its name, description, parameters, and return type. 
# This schema will allow the model to understand how to call these functions and what arguments to pass when it decides to use them during its reasoning process.
#(JSOn Schema tells the LLM how to use the tool)


# Difference2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.



# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(name="Ollama Chat", run_type="llm") #This decorator allows us to trace the execution of the ollama_chat_traced function and log it to Langsmith. By using this decorator, we can see when the function is called, what arguments are passed to it, and what value it returns. This can help us understand how the agent is interacting with the Ollama chat model and identify any issues or areas for improvement in its usage.Theis is is taken care of automatically when using LangChain's llm classes, but since we are calling ollama.chat() directly, we need to manually add this traceable function to ensure that our LLM calls are logged to Langsmith for monitoring and analysis.
def ollama_chat_traced(messages):
    
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop") #This decorator allows us to trace the execution of the run_agent function and log it to Langsmith. By using this decorator, we can see how the agent is making decisions, which tools it is calling, and how it is generating its responses. This can help us understand the agent's behavior and identify any issues or areas for improvement.This is similar to using LangChain's Agent class, which automatically traces the agent's reasoning process. Since we are implementing the agent loop manually, we need to add this traceable decorator to ensure that the entire loop is logged to Langsmith for monitoring and analysis.
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }



    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages) #This sends the list of messages to the Ollama chat model and gets a response back. The ollama_chat_traced function is a wrapper around the ollama.chat() function that includes tracing for Langsmith. The response variable will contain the model's response, which may include content (text) as well as tool calls (instructions to use tools). We can then analyze this response to see if it includes any tool calls or if it is the final answer to the user's question.
        ai_message = response.message #The message object from the model's response, which contains the content of the response as well as any tool calls that the model has decided to make based on its reasoning process. The ai_message variable will allow us to access both the text content of the model's response and the tool calls it wants to execute, which we can then process accordingly in our agent loop.

        tool_calls = ai_message.tool_calls #Which tools the model decided should be invoked. This works only if:1.You passed tools/functions to the model 2.Your model supports tool calling (e.g. GPT-4, some Ollama models with function calling) . 
        #The tool_calls variable will contain a list of tool calls that the model has decided to make based on its response. 
        # Each tool call will include the name of the tool to be called, the arguments to be passed to the tool, and a unique identifier for the tool call. 
        # We can use this information to execute the appropriate tools and provide the results back to the model in subsequent iterations.

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)


        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")


## In this code, we have implemented a simple agent loop that interacts with an Ollama chat model to answer a user's question about 
# product pricing and discounts. 
# The agent has access to two tools: get_product_price and apply_discount, which it can call during its reasoning process to retrieve
# information and perform calculations.
#  We have also added tracing using the @traceable decorator from Langsmith to log the execution of our functions and the agent loop 
# for monitoring and analysis purposes. 
# The agent will continue to iterate, calling tools as needed, until it generates a final answer without any tool calls or reaches 
# the maximum number of iterations.

# This implementtion is specific to Ollama and does not use LangChain's Agent or Tool classes, so we had to manually define the tools, their JSON schemas, and the agent loop.
# For claude or other models, the implementation would be different if we plan tio implement manually. 
# WE can use LangChain's Agent and Tool classes to abstract away much of this manual work and have a more standardized implementation that can work across different LLMs with minimal changes.

#Q & A
#1. In this file, tool schemas are defined as hand-written JSON dictionaries (tools_for_llm). What does this reveal about what LangChain's @tool decorator was doing for us?

# - @tool was auto-generating these exact JSON schemas from the function's name, type hints, and docstring, saving us from writing and maintaining them by hand.
# - Compare: with @tool, you write a typed Python function with a docstring, and the schema is generated automatically. Without it, you must manually specify each parameter's name, type, description, and required status in a nested JSON structure.

#2.   Instead of SystemMessage(content=...) and HumanMessage(content=...), this file uses plain dictionaries like {"role": "system", "content": ...}. What's the trade-off?

#- Dictionaries are provider-specific — this format works for Ollama/OpenAI but other providers may use different structures, whereas LangChain's message types provide a universal format
# The {"role": "...", "content": "..."} format follows the OpenAI/Ollama convention. If you switched to a provider with a different message format, you'd need to rewrite all your message construction. LangChain's typed messages abstract this away. 

#3. The LangChain version calls tools with tool_to_use.invoke(tool_args), but this file calls them with tool_to_use(**tool_args). What is the difference?

# - invoke() is LangChain's standardized execution interface that adds validation, tracing, and error handling around the call, while **tool_args is a direct Python function call with none of that.
# - LangChain's invoke() wraps the function call with input validation (checking types match the schema), integration with LangSmith tracing, and consistent error handling.
# - The raw **tool_args call is plain Python — if the arguments are wrong, you get a raw Python error with no additional context.

#4. The LangChain version uses tool_call.get("name") (dict access), but this file uses tool_call.function.name (attribute access). Why the difference?

# - LangChain returns tool calls as standardized dictionaries, while Ollama's client returns them as typed objects with nested attributes — each provider structures this data differently .
# - This illustrates a key point: without LangChain, your code is coupled to a specific provider's response format.
#  - Ollama returns objects with .function.name and .function.arguments. 
# - OpenAI's raw API returns a similar but not identical structure. 
# - LangChain normalizes all of these into a consistent dictionary format.

#5. In the LangChain version, ToolMessage requires a tool_call_id. In this raw version, the tool result is appended as {"role": "tool", "content": str(observation)} with no ID. Why?

# - Ollama's local models don't require a tool_call_id to match results to calls, while cloud providers like OpenAI strictly require it — this is another provider-specific difference that LangChain handles for you.
# - Ollama processes tool results sequentially and doesn't enforce ID matching. 
# - OpenAI's API strictly requires tool_call_id to correlate results with calls. 
# - LangChain's ToolMessage always requires the ID, ensuring your code works across all providers without modification.

#6. This file uses @traceable decorators for tracing, while the LangChain version uses init_chat_model() which integrates tracing automatically. What does this tell us?

# - Without LangChain, you must manually decorate each function with @traceable and specify run_type to get observability, whereas LangChain's model and tool wrappers report traces automatically .
# - Notice the extra effort: @traceable(run_type="tool") on each tool function, @traceable(name="Ollama Chat", run_type="llm") on the chat wrapper, and
# - @traceable(name="Ollama Agent Loop") on the agent. 
# - With LangChain, the @tool decorator and init_chat_model() handle tracing integration automatically.

#7. Comparing this file to the LangChain version, what fundamentally stays the same and what changes?

# - The agent loop pattern stays the same (iterate, check for tool calls, execute, append results), but without LangChain you must manually handle schemas, message formatting, provider-specific response parsing, tool invocation, and tracing.
# - The pattern is the same — that’s the agent loop concept. But the plumbing changes significantly: 35+ lines of manual JSON schemas, provider-specific dict messages, attribute-based response parsing, direct function calls without validation, and manual tracing decorators. LangChain eliminates this boilerplate so you can focus on the logic.











