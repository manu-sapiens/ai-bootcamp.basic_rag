# llm_image_analysis.py
# -------------------------- LOCAL ---------------------
from llm import query_llm

def analyze_image(image_url: str, question: str = "What's in this image?", max_tokens: int = 300) -> None:
    """
    Submits an image (via URL) along with a question to the LLM and prints the answer.
    
    Args:
        image_url (str): URL to the image to analyze.
        question (str): A text question about the image.
        max_tokens (int): Maximum tokens for the response.
    """
    # Build the message payload with two parts:
    # 1. A text prompt asking a question about the image.
    # 2. An image component providing the URL of the image.
    
    image_analyze_prompt = [
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
    ]
        
    # Call query_llm() with the prompt list.
    answer = query_llm(
        prompt=image_analyze_prompt,
        max_tokens=max_tokens
    )

    return answer
#

if __name__ == "__main__":
    # Example image URL (you can substitute with any publicly available image URL)
    test_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    
    # Ask the LLM to analyze the image.
    answer = analyze_image(test_image_url, "What's in this image?")
    print(answer)
#
