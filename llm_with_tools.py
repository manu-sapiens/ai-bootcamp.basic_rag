# llm_with_tools.py
# -------------------------- NATIVE --------------------
import json
import random
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
# Each tool definition includes the tool's name and a description.
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

If a tool is needed, respond with JSON in this exact format:
{{
  "tool": "tool_name",
  "params": {{"operation": "<add|multiply>", "numbers": [number, number]}}
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

# --- Test Harness for 100 Queries ---
def test_tool_detection_system(num_tests: int = 100):
    """
    Generate random queries for the calculator tool and compile statistics.
    
    Categories:
      - success: correct tool detection and calculation.
      - computation_failure: tool called but the computed result is wrong.
      - format_failure: tool detected but the returned JSON has wrong/missing parameters.
      - no_tool_failure: failure to call a tool when it should have been called.
    """
    success = 0
    computation_failure = 0
    format_failure = 0
    no_tool_failure = 0

    for _ in range(num_tests):
        # Randomly choose between addition and multiplication.
        op = random.choice(["add", "multiply"])
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        if op == "add":
            query = f"What is {a} plus {b}?"
            expected = a + b
        else:
            query = f"What is {a} multiplied by {b}?"
            expected = a * b

        # Use the tool detection system to get the JSON.
        tool_call = detect_tools(query, TOOLS)
        
        # If no tool was detected when one is needed, count as a failure.
        if not tool_call:
            no_tool_failure += 1
            print(f"[No Tool] Query: '{query}'")
            continue
        
        # Check for the proper JSON format.
        if not isinstance(tool_call, dict) or "tool" not in tool_call or "params" not in tool_call:
            format_failure += 1
            print(f"[Format Failure] Query: '{query}' | Response: {tool_call}")
            continue

        # Verify that the right tool is called.
        if tool_call["tool"] != "calculator":
            format_failure += 1
            print(f"[Wrong Tool] Query: '{query}' | Response: {tool_call}")
            continue

        params = tool_call["params"]
        # Validate that the necessary parameters are present.
        if not isinstance(params, dict) or "operation" not in params or "numbers" not in params:
            format_failure += 1
            print(f"[Params Format Failure] Query: '{query}' | Params: {params}")
            continue

        # Execute the calculator tool with the provided parameters.
        result = execute_tool("calculator", params)
        try:
            computed = float(result)
        except Exception:
            computation_failure += 1
            print(f"[Computation Error] Query: '{query}' | Result: {result}")
            continue

        # Compare the computed result with the expected value.
        if computed == expected:
            success += 1
            print(f"[{success}#] Query: '{query}' | Computed: {computed}")
        else:
            computation_failure += 1
            print(f"[Wrong Calculation] Query: '{query}' | Expected: {expected}, Got: {computed}")

    # Print a summary of test outcomes.
    print("----- Test Summary -----")
    print(f"Total tests: {num_tests}")
    print(f"Success: {success}")
    print(f"Computation failures (wrong result): {computation_failure}")
    print(f"Format failures (tool called, but wrong parameters): {format_failure}")
    print(f"Failures to call tool when needed: {no_tool_failure}")
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
    
    # # Example 3: Run the test harness with 10 generated queries.
    print("\nRunning 10-query test harness for tool detection and execution...")
    test_tool_detection_system(10)
