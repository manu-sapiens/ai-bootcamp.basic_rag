#test_tool.py
# -------------------------- NATIVE --------------------
import random
# -------------------------- LOCAL ---------------------
from llm_with_tools import detect_tools, execute_tool, TOOLS  # Your existing LLM module

# --- Test Harness for X Queries ---
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

    # # Example: Run the test harness with 10 generated queries.
    print("\nRunning 10-query test harness for tool detection and execution...")
    test_tool_detection_system(10)
#