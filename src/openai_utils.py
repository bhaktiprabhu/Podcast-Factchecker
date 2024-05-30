"""Module containing the OpenAI Summary Generator class."""

import os
from typing import List

import dotenv
from openai import AzureOpenAI, BadRequestError

dotenv.load_dotenv()

OPENAI_CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version="2024-02-01"
)

CHECKWORTHY_SYSTEM_PROMPT = """
You are a fact-checker system who responds only in one word either 'Yes' or 'No'.
Your task is to identify whether a given text in the english language os verifiable using a search \
engine in the context of fact-checking.
'Yes' means the text is a factual checkworthy statement.
'No' means that the text is not checkworthy, it might be an opinion, a question, or others.
"""

CHECHWORTHY_PROMPT = [
    {"role": "system", "content": CHECKWORTHY_SYSTEM_PROMPT},
    {"role": "user", "content": "I think Apple is a good company."},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Apple's CEO is Tim Cook."},
    {"role": "assistant", "content": "Yes"},
]

IDENTIFY_STANCE_SYSTEM_PROMPT = """
You are given a claim and an evidence text both in the english language.
You need to decide whether the evidence supports or refutes the claim.
Choose from the following two options.
SUPPORTS -> The evidence supports the claim.
REFUTES -> The evidence refutes the claim.
Pick the correct option either 'SUPPORTS' or 'REFUTES'. You must not add any other words.
"""

IDENTIFY_STANCE_PROMPT = [
    {"role": "system", "content": IDENTIFY_STANCE_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "[Claim]: India has the largest population in the world. \
    [Evidence]: In 2023 India overtook China to become the most populous country.",
    },
    {"role": "assistant", "content": "SUPPORTS"},
    {
        "role": "user",
        "content": "[Claim]: The Earth is flat. \
    [Evidence]: Scientific consensus and numerous observations from space confirm that the Earth is spherical.",
    },
    {"role": "assistant", "content": "REFUTES"},
]


class BadResponseError(Exception):
    """Custom error defined to handle OpenAI API response

    Args:
        Exception (_type_): Extends class Exception
    """


def generate_response(messages: List, model=os.getenv("AZURE_OPENAI_MODEL_ID")) -> str:
    """Generates a response from the OpenAI API for the given prompt.

    Args:
        messages: A list of example chat prompt
        model: The model to use for generating the response. Default is None.

    Returns:
        The API's response as a string.
    """
    response = OPENAI_CLIENT.chat.completions.create(model=model, messages=messages, max_tokens=10)
    return response.choices[0].message.content.strip()


def claim_detection(text_prompt: str) -> str:
    """Generate Open AI response for Checkworthy Prompt

    Args:
        text_prompt (str): text for verifying whether it is a claim or not

    Returns:
        str: The API's response as a string (Yes or No).
    """
    messages = CHECHWORTHY_PROMPT
    messages.append({"role": "user", "content": text_prompt})

    try:
        response = generate_response(messages)
        if response in ("Yes", "No"):
            return response
        else:
            raise BadRequestError
    except BadRequestError:
        print(text_prompt)
        return "No"


def stance_detection(claim: str, evidence_snippet: str) -> str:
    """Generate Open AI response for Identifying Stance

    Args:
        claim (str): claim text
        evidence_snippet (str): evidence snippet to for identifying stance

    Returns:
        str: The API's response as a string (SUPPORTS or REFUTES).
    """
    text_prompt = f"[Claim]: {claim} [Evidence]: {evidence_snippet}"
    messages = IDENTIFY_STANCE_PROMPT
    messages.append({"role": "user", "content": text_prompt})

    try:
        response = generate_response(messages)
        if response in ("SUPPORTS", "REFUTES"):
            return response
        else:
            raise BadResponseError
    except BadRequestError:
        print("OpenAI BadRequestError occured for claim: ", claim)
        return "REFUTES"
    except BadResponseError:
        print("OpenAI BadResponseError occured for claim: ", claim, "Response: ", response)
        return "REFUTES"
