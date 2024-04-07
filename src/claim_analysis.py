"""This module includes function for Claim Detection Analysis"""

import os

import numpy as np
import pandas as pd

import openai_utils
from factiverse_api import factiverse_api
from utils import execute_query_pandas, execute_query_sqlite


def get_segmentation_id(episode_id: int) -> int:
    """Get Segmentation Id for an Episode

    Args:
        episode_id (int): Episode Id

    Returns:
        int: segmentation Id
    """
    q = f"""
            select id
            from api_segmentation
            where transcription_id = (
                select id
                from api_transcription
                where item_id = {episode_id}
            )
            and name = 'Factiverse CW/MO/CS'
        """
    seg_id = execute_query_sqlite(q, 'one')[0]
    return seg_id


def get_utterances(episode_id: int) -> pd.DataFrame:
    """Fetch all utterances for an episode

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.DataFrame: Utterances for an Episode
    """
    seg_id = get_segmentation_id(episode_id)

    q = f"""
            select id, text, text_coref
            from api_utterance
            where segmentation_id = {seg_id}
        """

    utterances = execute_query_sqlite(q)
    utterance_df = pd.DataFrame(
        utterances, columns=["utterance_id", "text", "text_coref"]
    )
    utterance_df["text_coref"] = utterance_df["text_coref"].fillna(utterance_df["text"])
    return utterance_df


def get_factiverse_claim_df(episode_id: int) -> pd.DataFrame:
    """Gets Factiverse Claim Detection Results for an episode transcription

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.DataFrame: Factiverse Claim Detection Output for an Episode
    """
    utterance_df = get_utterances(episode_id)

    facti_claim_df = utterance_df[["utterance_id", "text"]]
    facti_claim_df.is_copy = False

    for i, text in enumerate(utterance_df["text_coref"]):
        facti_claim_check = factiverse_api.claim_detection(text)
        if len(facti_claim_check["detectedClaims"]) == 0:
            facti_claim_df.loc[i, "factiverse_claim"] = "Not Checkworthy"
        else:
            facti_claim_df.loc[i, "factiverse_claim"] = "Checkworthy"

    return facti_claim_df


def get_openai_claim_df(episode_id: int) -> pd.DataFrame:
    """Gets OpenAI Claim Detection Results for an episode transcription

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.DataFrame: OpenAI Claim Detection Output for an Episode
    """
    utterance_df = get_utterances(episode_id)

    openai_claim_df = utterance_df[["utterance_id", "text"]]
    openai_claim_df.is_copy = False

    for i, text in enumerate(utterance_df["text_coref"]):
        openai_claim_check = openai_utils.generate_response(text)
        if openai_claim_check in ('No', 'no'):
            openai_claim_df.loc[i, "openai_gpt4_claim"] = "Not Checkworthy"
        elif openai_claim_check in ('Yes', 'yes'):
            openai_claim_df.loc[i, "openai_gpt4_claim"] = "Checkworthy"

    return openai_claim_df


def get_annotation_claim_df(episode_id: int) -> pd.DataFrame:
    """Gets annotation results for an episode transcription

    Args:
        episode_id (_type_): Episode Id

    Returns:
        pd.DataFrame: Annotation claims for an episode
    """
    seg_id = get_segmentation_id(episode_id)
    q = f"""
            select
                utterance_id,
                sum(case when category = 'Checkable' then 1 else 0 end) as CW_claim_count,
                sum(case when category = 'Not Checkable' then 1 else 0 end) as Not_CW_claim_count
            from annotation_df
            where qualifier = 'Checkworthiness'
            and segmentation_id = {seg_id}
            group by utterance_id
            """
    annot_claim_df = execute_query_pandas(q)
    annot_claim_df["majority_claim"] = np.where(
        annot_claim_df["CW_claim_count"] < annot_claim_df["Not_CW_claim_count"],
        "Not Checkworthy",
        "Checkworthy",
    )

    annot_claim_df["utterance_id"] = annot_claim_df["utterance_id"].astype(int)

    return annot_claim_df


def generate_claim_comparison_csv(episode_id: int):
    """Generates csv for an episode with analysis

    Args:
        episode_id (int): Episode Id
    """

    facti_claim_df = get_factiverse_claim_df(episode_id)
    openai_claim_df = get_openai_claim_df(episode_id)
    annot_claim_df = get_annotation_claim_df(episode_id)

    result_df = pd.merge(pd.merge(facti_claim_df, openai_claim_df, on='utterance_id', how='left'),
                         annot_claim_df, on='utterance_id', how='left')

    result_df["CW_claim_count"] = result_df["CW_claim_count"].fillna(0)
    result_df["Not_CW_claim_count"] = result_df["Not_CW_claim_count"].fillna(0)

    result_df["CW_claim_count"] = result_df["CW_claim_count"].astype(int)
    result_df["Not_CW_claim_count"] = result_df["Not_CW_claim_count"].astype(int)

    result_df["output_matched"] = result_df["factiverse_claim"] == result_df["majority_claim"]

    result_df = result_df.sort_values(by='utterance_id')
    result_df = result_df.reset_index(drop=True)

    result_df['output_matched'] = np.where(
        result_df['majority_claim'].isnull(),
        None,
        result_df['output_matched']
    )

    ouput_file_name = f"episode_{episode_id}_claim_analysis.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ouput_file_path = os.path.join(current_dir, "output", ouput_file_name)

    result_df.to_csv(ouput_file_path, index=False)
