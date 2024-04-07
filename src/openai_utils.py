
"""Module containing the OpenAI Summary Generator class."""
import os

import dotenv
import requests
from openai import AzureOpenAI, BadRequestError

dotenv.load_dotenv()

OPENAI_CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"), 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)

CHECKWORTHY_PROMPT = """
You are a fact-checker system who responds only in one word either 'Yes' or 'No'. 
Your task is to identify whether a given text in the english language os verifiable using a search \
engine in the context of fact-checking.
'Yes' means the text is a factual checkworthy statement.
'No' means that the text is not checkworthy, it might be an opinion, a question, or others.
"""


def generate_response(text_prompt: str, model=os.getenv("AZURE_OPENAI_MODEL_ID")) -> str:
    """Generates a response from the OpenAI API for the given prompt.

    Args:
        prompt: The prompt to send to the API.
        model: The model to use for generating the response. Default is None.

    Returns:
        The API's response as a string.
    """
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": CHECKWORTHY_PROMPT},
                    {"role": "user", "content": "I think Apple is a good company."},
                    {"role": "assistant", "content": "No"},
                    {"role": "user", "content": "Apple's CEO is Tim Cook."},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": text_prompt}],
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except BadRequestError:
        print(text_prompt)
        return 'No'
        # raise requests.exceptions.HTTPError(
        #     f"Failed to obtain token: {response.status_code} {response.text}"
        # )
