"""This module includes all utility functions"""

import os
import sqlite3
from typing import List, Tuple, Union

import pandas as pd
from pandasql import sqldf

# Get path for current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# create connection to podcast database
db_file_path = os.path.join(current_dir, "data", "podcast_db.sqlite3")
sqliteConnection = sqlite3.connect(db_file_path)
cursor = sqliteConnection.cursor()

# Store annotation results in DataFrame
csv_file_path = os.path.join(current_dir, "data", "annotation_output.csv")
data = pd.read_csv(csv_file_path)
annotation_df = data[
    [
        "id",
        "qualifier",
        "category",
        "agent",
        "utterance_id",
        "text",
        "text_coref",
        "segmentation_id",
    ]
]


def execute_query_pandas(query: str) -> pd.DataFrame:
    """Executes sql query on Pandas DataFrame

    Args:
        query (str): SQl Query to Execute

    Returns:
        any: sql query result
    """
    result = sqldf(query)
    return result


def execute_query_sqlite(query: str, fetch_type: str = "all") -> Union[List, Tuple]:
    """Executes sql query on Sqlite DB

    Args:
        query (str): SQl Query to Execute
        fetch_type (str, optional): fetch 'one' or 'all' records.
                                    Defaults to 'all'
    Returns:
        Union[List, Tuple]: sql query result
    """
    cursor.execute(query)

    if fetch_type == "all":
        result = cursor.fetchall()
    elif fetch_type == "one":
        result = cursor.fetchone()
    else:
        raise ValueError("Invalid Fetch type")

    return result


def list_all_podcasts():
    """Lists all available podcasts in Database"""
    q = "select id, title from api_audiochannel"
    podcasts = execute_query_sqlite(q)

    for item in podcasts:
        print("Id: " + str(item[0]) + " || " + "Podcast: " + item[1])


def list_all_episodes(podcast_id: int = None) -> List:
    """Lists all episodes for a podcast/all podcasts.

    Args:
        podcast_id (int, optional): _description_. Defaults to None.

    Returns:
        List: List of Ids of available episodes in the database
    """
    q = """ select ep.id, ep.title, pod.title
            from api_audioitem ep
            inner join api_audiochannel pod on ep.channel_id = pod.id
        """
    if podcast_id is not None:
        q = q + f"where pod.id = {podcast_id}"

    episodes = execute_query_sqlite(q)

    ep_ids = []

    for item in episodes:
        ep_ids.append(item[0])
        print(
            "Id: "
            + str(item[0])
            + " || "
            + "Episode: "
            + item[1]
            + " || "
            + "Podcast: "
            + item[2]
        )

    return ep_ids
