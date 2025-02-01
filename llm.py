#llm_query.py
# -------------------------- NATIVE --------------------
import os
# -------------------------- REQUIREMENTS.TXT ----------
from dotenv import load_dotenv  # For loading .env file
from openai import OpenAI
# -------------------------- LOCAL ---------------------

# ------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

VERBOSE = False
MODEL = os.getenv("LLM_MODEL")  # Replace with your model name
OPENAI_API_KEY = os.getenv("API_KEY")  # Read API key from .env

# initialize the OpenAI client
client = OpenAI()
def query_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs  # Extra parameters such as temperature, max_tokens, etc.
) -> str:
    """
    Send a prompt to the LLM and return the response.

    Args:
        prompt (str): The input prompt.
        system_prompt (str): A system-level prompt.
        **kwargs: Additional keyword arguments passed to the LLM API call,
                  e.g., temperature, max_tokens, etc.

    Returns:
        str: The LLM's response.
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **kwargs  # Pass additional parameters into the API call
        )
        text = completion.choices[0].message.content
 
        if VERBOSE:
            print(completion.model_dump_json(indent=2))
            usage = dict(completion).get('usage')
            print(f"Usage: {usage}")
        #

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
