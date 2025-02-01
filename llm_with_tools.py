# llm_with_tools.py
# -------------------------- NATIVE --------------------
import json
from typing import Dict, Any, List
# -------------------------- LOCAL ---------------------
from llm import query_llm  # Your existing LLM module

# --- Example Tools ---
def calculator(operation: str, numbers: list) -> float:
    """A simple calculator tool for basic arithmetic."""
    try:
        if operation == "add":
            return sum(numbers)
        elif operation == "multiply":
            result = 1
            for num in numbers:
                result *= num
            return result
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    except Exception as e:
        return f"Error: {str(e)}"

# --- Tool Definitions ---
# Each tool definition can include the tool's name and a description.
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Supports 'add' and 'multiply'."
    },
    # You can add more tool definitions here.
]

# --- Tool Detection Logic ---
def detect_tools(user_query: str, tools: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Ask the LLM to detect if a tool is needed for the query.
    It now takes a list of tool definitions to guide the LLM.
    
    Returns a JSON with tool name and parameters (or an empty dict if no tool is needed).
    """
    # Create a formatted string that lists the available tools and their descriptions.
    tools_info = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])
    
    system_prompt = f"""
You are a tool detection system. Analyze the user's query and decide if one of the following tools is required:
{tools_info}

For the calculator tool, use this exact format:
{{
  "tool": "calculator",
  "params": {{
    "operation": "add|multiply",
    "numbers": [list_of_numbers]
  }}
}}

If no tool is needed, respond with an empty JSON object: {{}}
"""
    response = query_llm(
        prompt=user_query,
        system_prompt=system_prompt,
        temperature=0.0  # Force deterministic output
    )
    
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        # If the LLM response is not valid JSON, assume no tool is needed.
        return {}
    #
#

def execute_tool(tool_name: str, params: Dict) -> str:
    """Execute a tool and return its result as a string."""
    if tool_name == "calculator":
        return str(calculator(**params))
    else:
        return f"Error: Unknown tool '{tool_name}'."
    #    
#

def query_llm_with_tools(user_query: str) -> str:
    """
    Full pipeline: detect tools, execute if needed, and generate a final answer.
    """
    # Step 1: Detect if a tool is needed, providing the list of available tools.
    tool_call = detect_tools(user_query, TOOLS)
    if not tool_call:
        # No tool was indicated, so just query the LLM directly.
        return query_llm(user_query)
    
    # Step 2: Execute the tool.
    tool_result = execute_tool(tool_call["tool"], tool_call["params"])
    
    # Step 3: Generate the final answer using the tool's result.
    final_prompt = f"User Query: {user_query}\nTool Result: {tool_result}"
    return query_llm(final_prompt)
#

# --- Test Cases ---
if __name__ == "__main__":
    # Example 1: A query that should trigger the calculator tool.
    query = "What is 123 multiplied by 456?"
    print(f"Query: {query}")
    print(f"Final Answer: {query_llm_with_tools(query)}")
    
    # Example 2: A query that doesn't require any tool.
    query = "Explain quantum computing in simple terms."
    print(f"\nQuery: {query}")
    print(f"Final Answer: {query_llm_with_tools(query)}")
    
    # Example 3: generate 100 queries to test the tool detection system with the calculator tool.
    # and check the result each time
    # compile nb of success, computation failure (wrong result), format failure (tool called, but wrong parameters)
    # and failure to call a tool when needed
    # LEFT FOR THE STUDENTS TO DO
    
#
