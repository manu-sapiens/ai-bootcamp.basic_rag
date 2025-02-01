#llm_query.py
# -------------------------- NATIVE --------------------
import os
# -------------------------- REQUIREMENTS.TXT ----------
from dotenv import load_dotenv  # For loading .env file
from openai import OpenAI
# -------------------------- LOCAL ---------------------
# Load environment variables from .env file
load_dotenv()

VERBOSE = False
MODEL = os.getenv("LLM_MODEL")  # Replace with your model name
OPENAI_API_KEY = os.getenv("API_KEY")  # Read API key from .env

# initialize the OpenAI client
client = OpenAI()

def query_llm(prompt: str, system_prompt = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 100) -> str:
    """
    Send a prompt to the LLM and return the response.
    
    Args:
        prompt (str): The input prompt.
        temperature (float): Controls randomness (0 = deterministic, 1 = creative).
        max_tokens (int): Maximum length of the response.
    
    Returns:
        str: The LLM's response.
    """

    # Send the request
    try:
        
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},    
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        if VERBOSE: print(completion.model_dump_json(indent=2))

        text = completion.choices[0].message.content
        usage = dict(completion).get('usage')
        
        if VERBOSE: print(f"Usage: {usage}")
        if VERBOSE: print(f"Response: {text}")
        
        return text
    except Exception as e:
        return f"Error: {str(e)}"
    #
#

if __name__ == "__main__":
    """Interactive loop for querying the LLM."""
    print("LLM Query Tool (type 'exit' to quit)")
    while True:
        # Get user input
        prompt = input("YOU: ")
        if prompt.lower() == "exit":
            break

        # Query the LLM
        response = query_llm(prompt)
        print(f"LLM: {response}")
#
