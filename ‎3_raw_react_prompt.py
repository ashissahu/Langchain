# CHANGE 1: Add re + inspect — we'll parse tool calls from raw text instead of structured JSON.
import re
import inspect # For introspecting tool function signatures and docstrings
from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# CHANGE 2: Tools are just a dict of functions, no LangChain agent or tool management. The LLM has no idea these are "tools" — they're just functions we can call in Python. The prompt is what gives them meaning and structure.
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

# Introspect tool functions to generate descriptions for the prompt. This allows us to include the function signature (parameters) and docstring (description) for each tool in the prompt, which helps the LLM understand how to use them. We use __wrapped__ to get the original function in case it's wrapped by a decorator like @traceable, so we can show the correct signature and docstring rather than the generic one from the wrapper.
def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function) # Get the original function if it's wrapped by a decorator. Here we want the original signature and docstring, not the wrapper added by @traceable. Here  the original function is the one we defined with the actual parameters (e.g., product: str), while the wrapper added by @traceable has a generic signature (*args, **kwargs). By accessing __wrapped__, we can get the original function's signature and docstring for accurate tool descriptions in the prompt.
        signature = inspect.signature(original_function) # Get the function signature (e.g., (product: str)) for the original function. This allows us to show the correct parameters in the prompt, rather than the generic (*args, **kwargs) signature of the wrapper added by @traceable.
        docstring = inspect.getdoc(tool_function) or "" # Get the docstring for the tool function, which describes what the tool does. This will be included in the prompt to help the LLM understand how to use the tool.
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions) # Join all tool descriptions into a single string with each tool on a new line. This will be included in the prompt to provide a clear list of available tools and how to use them.

tool_descriptions = get_tool_descriptions(tools) # Generate the tool descriptions for the prompt by introspecting the tool functions. This creates a nicely formatted list of tools with their parameters and descriptions, which we will include in the prompt to guide the LLM in how to use them.
print("*" * 20)
print("Tool Descriptions for Prompt:")
print(tool_descriptions)
print("*" * 20)

#path for prompt which was basic for langchian -https://smith.langchain.com/hub/hwchase17/react?organizationId=a7f8bd85-b797-5733-8038-ce1ba498c5e8

tool_names = ", ".join(tools.keys()) # Generate a comma-separated list of tool names for the prompt, which we will use to tell the LLM which tools are available for it to call. This is important for the LLM to know what actions it can take when answering the question.

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""




# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.

# We keep the @traceable wrapper around the LLM call so we can see it in LangSmith, but we have to change the signature to match how we're calling it now (with messages and options instead of tools).
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options): # options is a dict of generation options, e.g., {"stop": ["\nObservation"], "temperature": 0}
    return ollama.chat(model=model, messages=messages, options=options)





# --- Agent Loop ---

# The main agent loop. We send the entire prompt + history as one string to the LLM each time, and we parse its raw text output with regex to determine if it wants to call a tool or if it has provided a final answer. This is more fragile than structured messages, but it fits the "raw prompt" requirement of this exercise and shows how you can implement an agent loop without any special LangChain features — just prompt engineering and parsing LLM output.
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)


    # CHANGE 5: One prompt string replaces the system/user message split.
    # We use .format to inject the question into the prompt template, and we start with an empty scratchpad for the Thought/Action/Observation history that will grow with each iteration. This full prompt (template + history) is what we send to the LLM each time, and we parse its raw text output to determine the next steps.
    prompt = react_prompt.format(question=question)
    print(f"Initial Prompt:\n{prompt}")
    print("-" * 60)
    scratchpad = "" # This will accumulate the Thought/Action/Observation history as a growing string that we re-send to the LLM each iteration. This replaces the messages list we had before, which was easier to manage but doesn't fit the "raw prompt" requirement of this exercise.

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        print(f"Sending prompt to LLM:\n{full_prompt}")

        # Stop token prevents the LLM from generating its own Observation —
        # we inject the real tool result instead.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        # We look for "Final Answer: ..." in the LLM output to determine if it has completed its reasoning and is providing a final answer to the original question. If we find this, we extract the final answer and return it, ending the agent loop. If not, we look for "Action: ..." and "Action Input: ..." to determine if the LLM wants to call a tool, and we parse those out to execute the tool and get an observation, which we then add to the scratchpad for the next iteration.
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output) # Use regex to find "Final Answer: ..." .re.search(r"Final Answer:\s*(.+)", output) looks for the pattern "Final Answer:" followed by any text (the actual answer) and captures that text in a group. If this pattern is found in the LLM output, final_answer_match will be a match object containing the captured answer; if not found, it will be None. This allows us to determine if the LLM has provided a final answer and to extract it for returning at the end of the agent loop.\s* allows for any amount of whitespace between "Final Answer:" and the actual answer, and (.+) captures the rest of the line as the final answer.
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip() # If a final answer is found, extract it from the regex match group, strip any leading/trailing whitespace, and prepare to return it. This means the LLM has indicated it has completed its reasoning and is providing a final answer to the original question, so we can exit the loop and return this answer.
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer



        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output) # Use regex to find "Action: ..." in the LLM output. This looks for the pattern "Action:" followed by any text (the tool name) and captures that text in a group. If this pattern is found, action_match will be a match object containing the captured tool name; if not found, it will be None. This allows us to determine if the LLM is trying to call a tool and to extract which tool it wants to call.
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        # Extract the tool name and input from the regex matches, stripping any whitespace. This gives us the name of the tool the LLM wants to call and the raw input it wants to pass to that tool, which we will then process and execute in Python.
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")

        # Check if the tool exists and execute it with the parsed arguments. If the tool name from the LLM output doesn't match any of our available tools, we return an error observation. Otherwise, we call the corresponding function from our tools dict with the parsed arguments and convert the result to a string to use as the observation. This is where we actually execute the tool that the LLM has decided to call based on its reasoning in the prompt.
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        # We add the new Thought/Action/Observation to the scratchpad string, which accumulates the history of the agent's reasoning and tool calls. This full history is included in the prompt we send to the LLM each iteration, allowing it to see the entire context of what has happened so far. This replaces the more structured messages list we had before, and it relies on careful prompt engineering and formatting to ensure the LLM can understand and use this history effectively.
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")

# The expected flow is:
# 1. LLM receives the prompt with the question and tool descriptions.
# 2. LLM thinks: "I need to get the price of a laptop first."
# 3. LLM outputs an Action to call get_product_price with "laptop".
# 4. We parse that output, call get_product_price("laptop") in Python, and get the price (1299.99).
# 5. We add the Observation with the price back to the scratchpad and re-send the full prompt + history to the LLM.
# 6. LLM now thinks: "I have the price, now I need to apply the gold discount."
# 7. LLM outputs an Action to call apply_discount with the price and "gold".
# 8. We parse that output, call apply_discount(1299.99, "gold") in Python, and get the discounted price (1002.99).
# 9. We add the Observation with the discounted price back to the scratchpad and re-send the full prompt + history to the LLM.
# 10. LLM now thinks: "I have the final price after discount, I can provide the final answer."
# 11. LLM outputs a Final Answer with the discounted price.




#Here we have implemented a raw prompt-based agent loop without using any of LangChain's structured agent or tool management features. 
# The LLM is treated as a black box that generates text based on the prompt we give it, and we use regex parsing to interpret its 
# output and determine when to call tools and when it has provided a final answer. This approach is more fragile than using 
# structured messages and tool calls, but it demonstrates how you can implement an agent loop with just prompt engineering and 
# output parsing.