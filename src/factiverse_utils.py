"""This module includes functions for executing Factiverse APIs requests"""

import os
from typing import Dict

import dotenv
import requests

DEFAULT_ACCESS_TOKEN_URL = "https://factiverse-dev.eu.auth0.com/oauth/token"


class FactiverseAPI:
    """Factiverse API Class"""

    def __init__(self) -> None:
        """Initializes the Factiverse API client."""
        dotenv.load_dotenv()
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.access_token = self.get_access_token()

    def get_access_token(self, access_token_url: str = DEFAULT_ACCESS_TOKEN_URL) -> str:
        """Gets access token for running Factiverse APIs

        Args:
            access_token_url (str, optional):  Defaults to DEFAULT_ACCESS_TOKEN_URL.

        Returns:
            str: Access token
        """
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = requests.post(access_token_url, data=payload, timeout=10)

        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            raise requests.exceptions.HTTPError(f"Failed to obtain token: {response.status_code} {response.text}")

    def claim_detection(self, text: str, lang: str = "en", score_th: float = 0.5) -> Dict:
        """Factiverse Claim Detection API

        Args:
            text (str):
            lang (str, optional): Language. Defaults to "en"
            score_th(float, optional): threshold for claim detection.
                                    Defaults to 0.5

        Returns:
            Dict: Response from claim detection API
        """
        api_endpoint = "https://dev.factiverse.ai/v1/claim_detection"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

        payload = {
            "text": text,
            "lang": lang,  # Language code
            "claimScoreThreshold": score_th,
        }

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            print("Error:", response.status_code, response.text)
            return None

    def stance_detection(self, claim: str) -> Dict:
        """Factiverse Stance Detection API

        Args:
            text (str): text for verifying whether it is a claim or not
            access_token (str): generated access_token

        Returns:
            Dict: Response from stance detection API
        """
        api_endpoint = "https://dev.factiverse.ai/v1/stance_detection"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

        payload = {
            "claim": claim,
        }

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=40)

        if response.status_code == 200:
            return response.json()
        else:
            print("Error:", response.status_code, response.text)
            return None


factiverse_api = FactiverseAPI()
