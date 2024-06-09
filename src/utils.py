"""This module includes all general utility functions"""

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
csv_file_path = os.path.join(current_dir, "data", "toloka_annotation_output.csv")
toloka_data = pd.read_csv(csv_file_path)
toloka_annotation_df = toloka_data[
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
        podcast_id (int, optional): Podcast Id. Defaults to None.

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
        print("Id: " + str(item[0]) + " || " + "Episode: " + item[1] + " || " + "Podcast: " + item[2])

    return ep_ids


def get_episode_info(episode_id: int) -> Tuple:
    """Returns Episode Title and Podcast Title for an episode.

    Args:
        episode_id (int): Episode Id.

    Returns:
        Tuple: Episode and Podcast Title
    """
    q = f"""select ep.title, pod.title, pod.author, pod.categories
            from api_audioitem ep
            inner join api_audiochannel pod on ep.channel_id = pod.id
            where ep.id = {episode_id}
        """
    ep_info = execute_query_sqlite(q, "one")

    return ep_info


def get_podcast_dir_path(output_path: str, episode_id: int) -> str:
    """Returns podcast directory path for a given epidode id in given output path

    Args:
        output_path(str): Path for Podcast directory existence
        episode_id (int): Episode Id

    Returns:
        str: Directory Path for Podcast
    """
    q = f"""select pod.id, pod.title
            from api_audioitem ep
            inner join api_audiochannel pod on ep.channel_id = pod.id
            where ep.id = {episode_id}
        """
    podcast = execute_query_sqlite(q, "one")
    podcast_dir_name = (str(podcast[0]) + "_" + podcast[1]).replace(" ", "_")

    podcast_dir_path = os.path.join(output_path, podcast_dir_name)

    if not os.path.exists(podcast_dir_path):
        os.makedirs(podcast_dir_path)

    return podcast_dir_path


def create_csv_from_df(df: pd.DataFrame, episode_id: int, output_path: str, output_file_name: str):
    """Creates csv file from given dataframe in given folder path

    Args:
        df (pd.DataFrame): Dataframe for csv generation
        episode_id (int): Episode Id
        output_path (str): Output Directory Path
        output_file_name (str): Output File Name
    """
    podcast_dir = get_podcast_dir_path(output_path, episode_id)
    output_file_path = os.path.join(podcast_dir, output_file_name)

    df.to_csv(output_file_path, index=False)

    print(f"Output generated for Episode {episode_id} at {output_file_path}")
